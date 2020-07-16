# https://github.com/CalciferZh/SMPL
import torch
import torch.nn as nn
import pickle
import numpy as np
import scipy.sparse


class SMIL(nn.Module):
    def with_zeros(self, x):
        """
        Append a [0, 0, 0, 1] vector to a batch of [3, 4] matrices.

        Parameter:
        ---------
        x: Tensor to be appended of shape [N, 3, 4]

        Return:
        ------
        Tensor after appending of shape [N, 4, 4]

        """
        ret = torch.cat([x, self.e4.expand(x.shape[0], 1, -1)], dim=1)
        return ret

    def pack(self, x):
        """
        Append zero tensors of shape [4, 3] to a batch of [4, 1] shape tensors.

        Parameter:
        ----------
        x: A tensor of shape [batch_size, 4, 1]

        Return:
        ------
        A tensor of shape [batch_size, 4, 4] after appending.

        """
        ret = torch.cat(
            (torch.zeros((x.shape[0], x.shape[1], 4, 3), dtype=x.dtype, device=x.device), x),
            dim=3
        )
        return ret

    def rodrigues(self, r):
        """
        Rodrigues' rotation formula that turns axis-angle tensor into rotation
        matrix in a batch-ed manner.

        Parameter:
        ----------
        r: Axis-angle rotation tensor of shape [N, 1, 3].

        Return:
        -------
        Rotation matrix of shape [N, 3, 3].
        """
        theta = torch.norm(r, dim=(1, 2), keepdim=True)
        # avoid division by zero
        torch.max(theta, theta.new_full((1,), torch.finfo(theta.dtype).eps), out=theta)
        #The .tiny has to be uploaded to GPU, but self.regress_joints is such a big bottleneck it is not felt.

        r_hat = r / theta
        z_stick = torch.zeros_like(r_hat[:, 0, 0])
        m = torch.stack(
            (z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
             r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
             -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick), dim=1)
        m = m.reshape(-1, 3, 3)

        dot = torch.bmm(r_hat.transpose(1, 2), r_hat)  # Batched outer product.
        # torch.matmul or torch.stack([torch.ger(r, r) for r in r_hat.squeeze(1)] works too.
        cos = theta.cos()
        R = cos * self.eye + (1 - cos) * dot + theta.sin() * m
        return R

    def __init__(self, model_path='./model.pkl', sparse=True):
        super().__init__()

        self.parent = None
        self.model_path = None
        if model_path is not None:
            with open(model_path, 'rb') as f:
                self.model_path = model_path
                params = pickle.load(f)
                # The first three can be added simply:
                registerbuffer = lambda name: self.register_buffer(name,
                                                                   torch.as_tensor(params[name]))
                registerbuffer('weights')
                registerbuffer('posedirs')
                registerbuffer('v_template')
                registerbuffer('shapedirs')

                # Now for the more difficult...:
                # We have to convert f from uint32 to int32. (This is the indexbuffer)
                self.register_buffer('f', torch.as_tensor(params['f'].astype(np.int32)))
                self.register_buffer('kintree_table', torch.as_tensor(params['kintree_table'].astype(np.int32)))

                # J_regressor is a sparse tensor. This is (experimentally) supported in PyTorch.
                J_regressor = params['J_regressor']
                if scipy.sparse.issparse(J_regressor):
                    # If tensor is sparse (Which it is with SMPL/SMIL)
                    J_regressor = J_regressor.tocoo()
                    J_regressor = torch.sparse_coo_tensor([J_regressor.row, J_regressor.col],
                                                          J_regressor.data,
                                                          J_regressor.shape)
                    if not sparse:
                        J_regressor = J_regressor.to_dense()
                else:
                    J_regressor = torch.as_tensor(J_regressor)
                self.register_buffer('J_regressor', J_regressor)

                self.register_buffer('e4', self.posedirs.new_tensor([0, 0, 0, 1]))  # Cache this. (Saves a lot of time)
                self.register_buffer('eye', torch.eye(3, dtype=self.e4.dtype, device=self.e4.device))  # And this.
                self.set_parent()

        # Make sure the tree map is reconstructed if/when model is loaded.
        self._register_state_dict_hook(self.set_parent)

    def set_parent(self, *args, **kwargs):
        # Get kintree_table from state dict.
        # Make kinematic tree relations.
        id_to_col = {self.kintree_table[1, i].item(): i for i in range(self.kintree_table.shape[1])}
        self.parent = {
            i: id_to_col[self.kintree_table[0, i].item()]
            for i in range(1, self.kintree_table.shape[1])
        }
        # Must return None, since only a state dict or None return value is permitted for state dict hooks.


    def save_obj(self, verts, obj_mesh_name):
        with open(obj_mesh_name, 'w') as fp:
            for v in verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

            for f in self.f:  # Faces are 1-based, not 0-based in obj files
                fp.write('f %d %d %d\n' % (f[0] + 1, f[1] + 1, f[2] + 1))

    def regress_joints(self, vertices):
        """The J_regressor matrix transforms vertices to joints."""
        # Given the template + pose blend shapes.
        batch_size = vertices.shape[0]

        # We could get the result as torch.matmul(self.J_regressor, vertices) or
        #  torch.stack([self.J_regressor.mm(verts) for verts in vertices]) in case J_regressor is sparse.
        # But turns out there is a solution faster than both of the above:
        batch_vertices = vertices.transpose(0, 1).reshape(self.J_regressor.shape[1], -1)
        batch_results = self.J_regressor.mm(batch_vertices)
        batch_results = batch_results.reshape(self.J_regressor.shape[0], batch_size, -1).transpose(0, 1)
        return batch_results

    def rotate_translate(self, rotation_matrix, translation):
        transform = torch.cat((rotation_matrix, translation.unsqueeze(2)), 2)
        return self.with_zeros(transform)

    def forward(self, beta, pose, trans=None, simplify=False):
        """This module takes betas and poses in a batched manner.
        A pose is 3 * K + 3 (= self.kintree_table.shape[1] * 3) parameters, where K is the number of joints.
        A beta is a vector of size self.shapedirs.shape[2], that parameterizes the body shape.
        Since this is batched, multiple betas and poses should be concatenated along zeroth dimension.
        See http://files.is.tue.mpg.de/black/papers/SMPL2015.pdf for more info.
        """
        batch_size = beta.shape[0]  # Size of zeroth dimension.

        # The body shape is decomposed with principal component analysis from many subjects,
        #  where self.v_template is the average value. Then shapedirs is a subset of the orthogonal directions, and
        #  a the betas are the values when the subject is projected onto these. v_shaped is the "restored" subject.
        v_shaped = torch.tensordot(beta, self.shapedirs, dims=([1], [2])) + self.v_template

        # We turn the rotation vectors into rotation matrices.
        R_cube = self.rodrigues(pose.reshape(-1, 1, 3)).reshape(batch_size, -1, 3, 3)
        J = self.regress_joints(v_shaped)  # Joints in T-pose (for limb lengths)

        if not simplify:
            # Add pose blend shapes. (How joint angles morphs the surface)
            # Now calculate how joints affects the body shape.
            lrotmin = R_cube[:, 1:] - self.eye
            lrotmin = lrotmin.reshape(batch_size, -1)
            v_shaped += torch.tensordot(lrotmin, self.posedirs, dims=([1], [2]))

        # Now we have the un-posed body shape. Convert to homogeneous coordinates.
        rest_shape_h = torch.cat((v_shaped, v_shaped.new_ones(1).expand(*v_shaped.shape[:-1], 1)), 2)

        G = [self.rotate_translate(R_cube[:, 0], J[:, 0])]
        for i in range(1, self.kintree_table.shape[1]):
            G.append(
                torch.bmm(
                    G[self.parent[i]],
                    self.rotate_translate(R_cube[:, i], J[:, i] - J[:, self.parent[i]])))
        G = torch.stack(G, 1)
        Jtr = G[..., :4, 3].clone()
        G = G - self.pack(torch.matmul(G, torch.cat([J, J.new_zeros(1).expand(*J.shape[:2], 1)], dim=2).unsqueeze(-1)))

        # T = torch.tensordot(self.weights, G, dims=([1], [1]))
        # v = T.reshape(-1, 4, 4).bmm(rest_shape_h.reshape(-1, 4, 1)).reshape(batch_size, -1, 4)

        # Two next lines are a memory bottleneck.
        T = torch.tensordot(G, self.weights, dims=([1], [1])).permute(0, 3, 1, 2)

        v = torch.matmul(T, torch.reshape(rest_shape_h, (batch_size, -1, 4, 1))).reshape(batch_size, -1, 4)

        if trans is not None:
            trans = trans.unsqueeze(1)
            v[..., :3] += trans
            Jtr[..., :3] += trans

        return v, Jtr

def time_numpy(body_model, poses):
    return [body_model.set_params(pose.pose, pose.beta) for pose in poses]

def time_pytorch(body_model, betas, poses):
    for i in range(100):
        v = body_model(vbeta, vpose)

if __name__ == '__main__':
    from smil_np import SMILModel
    import timeit

    # The best configurations are:
    #  device = cuda, dtype = half, sparse = false
    #  device = cuda, dtype = float, sparse = true

    device = torch.device('cuda')  # torch.device('cuda')
    dtype = torch.float
    sparse = True if dtype is not torch.half else False # sparse Half Tensors are not supported (yet).

    SMILNP = SMILModel('./model.pkl')
    SMILPY = SMIL('./model.pkl', sparse=sparse).to(device)
    SMILPY = SMILPY.to(device, dtype, non_blocking=True)

    from file_utils import *
    poses = find_mini_rgbd(os.path.join('MINI-RGBD_web'))
    poses = [MicroRGBD(pose) for pose in poses[:4000]]  # If there is a memory error reduce size here.

    vbeta = torch.tensor(np.array([pose.beta for pose in poses])).to(device, dtype, non_blocking=True)
    vpose = torch.tensor(np.array([pose.pose for pose in poses])).to(device, dtype, non_blocking=True)
    vtrans = torch.tensor(np.array([pose.trans for pose in poses])).to(device, dtype, non_blocking=True)
    v, Jtr = SMILPY(vbeta, vpose, vtrans)  # Do the thing.
    time_pytorch(SMILPY, vbeta, vpose)

    # SMILNP.set_params(poses[i].pose, poses[i].beta, poses[i].trans)
    # print(Jtr.cpu()[i].numpy() - SMILNP.Jtr, Jtr[i].shape, SMILNP.Jtr.shape)  # See if there are any rounding errors.

    #with torch.cuda.profiler.profile() as prof:
    #    SMILPY(vbeta, vpose)  # Warmup CUDA memory allocator and profiler
    #    with torch.autograd.profiler.emit_nvtx():
    #        SMILPY(vbeta, vpose)

