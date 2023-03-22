import torch
import numpy as np


class LatticeQuantization:
    def __init__(self, args, hex_mat = np.array([[np.sqrt(3) / 2, 0], [1 / 2, 1]])):
        self.gamma = args.gamma
        self.overloading_vec = []
        # lattice generating matrix
        gen_mat = hex_mat/np.linalg.det(hex_mat)
        self.gen_mat = torch.from_numpy(gen_mat).to(torch.float32).to(args.device)

        # estimate P0_cov
        self.delta = (2 * args.gamma) / (2 ** args.R + 1)
        self.egde = args.gamma - (self.delta / 2)
        orthog_domain_dither = np.random.uniform(low=-self.delta / 2, high=self.delta / 2, size=[2, 1000])

        lattice_domain_dither = np.matmul(gen_mat, orthog_domain_dither)
        self.P0_cov = np.cov(lattice_domain_dither)

    def print_overloading_vec(self):
        sum = 0
        len = 0
        for item in self.overloading_vec:
            sum += item[0]
            len += item[1][0] * item[1][1]
        self.overloading_vec = []
        return (sum * 100 / len)

    def calc_overloading_vec(self, q_orthogonal_space):
        overloading = torch.zeros_like(q_orthogonal_space)
        overloading[q_orthogonal_space >= self.egde] = 1
        overloading[q_orthogonal_space <= -self.egde] = 1
        overloading_sum = float(torch.sum(torch.sum(overloading, dim=-1), dim=0))
        overloading_shape = q_orthogonal_space.shape
        # overloading = (float(torch.sum(torch.sum(overloading, dim=-1), dim=0)) / int(q_orthogonal_space.shape[0] * q_orthogonal_space.shape[1]))* 100
        self.overloading_vec.append([overloading_sum, overloading_shape])

    def __call__(self, input_vec):
        dither = torch.zeros_like(input_vec, dtype=input_vec.dtype)
        dither = torch.matmul(self.gen_mat, dither.uniform_(-self.delta / 2, self.delta / 2))  # generate dither

        input_vec = input_vec + dither

        # quantize
        orthogonal_space = torch.matmul(torch.inverse(self.gen_mat), input_vec)
        q_orthogonal_space = self.delta * torch.round(orthogonal_space / self.delta)
        self.calc_overloading_vec(q_orthogonal_space)
        q_orthogonal_space[q_orthogonal_space >= self.egde] = self.egde
        q_orthogonal_space[q_orthogonal_space <= -self.egde] = -self.egde
        input_vec = torch.matmul(self.gen_mat, q_orthogonal_space)

        return input_vec - dither


class ScalarQuantization:
    def __init__(self, args):
        # quantization levels spacing
        self.delta = (2 * args.gamma) / (2 ** args.R + 1)
        self.egde = args.gamma - (self.delta/2)

    def __call__(self, input):
        # decode
        dither = torch.zeros_like(input, dtype=torch.float32)
        dither = dither.uniform_(-self.delta / 2, self.delta / 2)  # generate dither
        input = input + dither

        # mid-tread
        q_input = self.delta * torch.round(input / self.delta)
        q_input[q_input >= self.egde] = self.egde
        q_input[q_input <= -self.egde] = -self.egde

        return q_input - dither