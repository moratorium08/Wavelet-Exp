# coding:utf-8
import unittest
from wavelet import serialize, deserialize
from vector import *
from compress import _compress, chunk_size, _decompress


class TestWavelet(unittest.TestCase):

    # TODO
    def test_serialize(self):
        flags = [True, False, True, False]
        #serd = serialize([range(8),range(4),range(2)], range(2), flags)

        #self.assertEqual(serd, [0, 1, 2, 3, 4, 5, 6, 7, 0, 1])
        #self.assertEqual(deserialize(serd, 3, 16, flags)[0][0],
                         #[0, 1, 2, 3, 4, 5, 6, 7])


class TestCompress(unittest.TestCase):

    def test_sub_compress(self):
        SIZE = 16
        K = 3
        h = [0 for i in range(chunk_size)]
        h[0] = np.float16(1 / sqrt(2))
        h[1] = np.float16(1 / sqrt(2))

        g = [0 for i in range(chunk_size)]
        g[0] = np.float16(1 / sqrt(2))
        g[1] = np.float16(-1 / sqrt(2))
        g_ = g[:SIZE]
        h_ = h[:SIZE]

        u = [i % 2 for i in range(16)]
        ret, flags = _compress(u, g_, h_, SIZE, K)
        u2 = _decompress(ret, flags, g_, h_, SIZE, K)
        self.assertEqual(map(int, map(round, u2)), u)

vec1 = [2, 1, 2, 1]
vec2 = [1, 2, 3, 4]


class TestVector(unittest.TestCase):

    def test_convolution_period(self):
        self.assertEqual(convolution_period(vec1, vec2), [14, 16, 14, 16])

    def test_downsampling(self):
        self.assertEqual(downsampling(upsampling([1, 2, 3])), [1, 2, 3])

    def test_bytes2vec(self):
        self.assertEqual(bytes2vec(b'\x01\x00\x02\x00\x03\x00'), [1, 2, 3])

    def test_vec2bytes(self):
        self.assertEqual(bytes2vec(vec2bytes([1, 2, 3])), [1, 2, 3])

    def test_add(self):
        self.assertEqual(add([1, -2, 3], [1, 1, 1]), [2, -1, 4])

    def test_sub(self):
        self.assertEqual(sub([1, -2, 3], [1, 1, 1]), [0, -3, 2])

    def test_norm(self):
        self.assertEqual(int(norm([3, 4])), 5)

    def test_reverse(self):
        self.assertEqual(reverse([1, 2, 3]), [1, 3, 2])

    def test_cycling(self):
        self.assertEqual(cycling([1, 2, 3, 4], 2), [1, 2, 1, 2, 1, 2, 1, 2])

    def test_upsampling(self):
        self.assertEqual(upsampling([1, 2, 3], 2), [1, 0, 2, 0, 3, 0])
        self.assertEqual(upsampling([1, 2, 3], 1), [1, 2, 3])

    def test_downsampling(self):
        self.assertEqual(downsampling([1, 0, 2, 0, 3, 0], 2), [1, 2, 3])
        self.assertEqual(downsampling([1, 2, 3, 4, 5, 6], 3), [1, 4])
        self.assertEqual(downsampling([1, 2, 3], 1), [1, 2, 3])

    def test_convolution_box(self):
        ret = convolution_box([1, 2, 3, 4], [1, 1, 1, 1], 2)
        self.assertEqual(map(int, ret), [1, 2])

    def test_sum_vec(self):
        self.assertEqual(sum_vec([[1,2,3], [1,1,1], [-1, -2, -3]]), [1, 1, 1])

    def test_level1_haarwavelet(self):
        h = [0 for i in range(4096)]
        h[0] = np.float16(1 / sqrt(2))
        h[1] = np.float16(1 / sqrt(2))

        g = [0 for i in range(4096)]
        g[0] = np.float16(1 / sqrt(2))
        g[1] = np.float16(-1 / sqrt(2))
        u = [0, 1, 2, 3, 4, 5]
        phi = h[:6]
        psi = g[:6]

        phi_rev = reverse(phi)
        psi_rev = reverse(psi)

        p = convolution_period(phi_rev, u)
        p = upsampling(downsampling(p))
        p = convolution_period(phi, p)

        q = convolution_period(psi_rev, u)
        q = upsampling(downsampling(q))
        q = convolution_period(psi, q)

        u2 = add(p, q)
        self.assertTrue(norm(sub(u, u2)) < 0.01)


if __name__ == '__main__':
    unittest.main(verbosity=2)
