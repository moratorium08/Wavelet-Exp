# coding:utf-8
import unittest
from wavelet import serialize, deserialize
from vector import *

class TestWavelet(unittest.TestCase):

    def test_serialize(self):
        flags = [True, False, True, False]
        serd = serialize([range(8),range(4),range(2)], range(2), flags)

        self.assertEqual(serd, [0, 1, 2, 3, 4, 5, 6, 7, 0, 1])
        self.assertEqual(deserialize(serd, 3, 16, flags)[0][0],
                         [0, 1, 2, 3, 4, 5, 6, 7])

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

vec1 = [2, 1, 2, 1]
vec2 = [1, 2, 3, 4]
class TestVector(unittest.TestCase):

    def test_convolution_period(self):
        self.assertEqual(convolution_period(vec1, vec2), [14, 16, 14, 16])

    def test_downsampling(self):
        self.assertEqual(downsampling(upsampling([1, 2, 3])), [1, 2, 3])

    def test_vec2bytes(self):
        self.assertEqual(bytes2vec(vec2bytes([1, 2, 3])), [1, 2, 3])

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
    unittest.main()
