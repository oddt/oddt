import os
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal
import oddt
from oddt.shape import usr, usr_cat, electroshape, usr_similarity

test_data_dir = os.path.dirname(os.path.abspath(__file__))
benzene_sdf = """
     RDKit          3D

  6  6  0  0  0  0  0  0  0  0999 V2000
   -1.0971    0.8220    0.2458 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.1708    1.3788    0.0966 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.2678    0.5568   -0.1492 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.0971   -0.8220   -0.2458 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.1708   -1.3788   -0.0966 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2678   -0.5568    0.1491 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  0
  2  3  1  0
  3  4  2  0
  4  5  1  0
  5  6  2  0
  6  1  1  0
M  END

"""
methylo_sdf = """
     RDKit          3D

  7  7  0  0  0  0  0  0  0  0999 V2000
    0.2879   -1.1465    0.0728 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.0997   -1.1111    0.1887 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.7974    0.0457   -0.1502 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.1081    1.1672   -0.6050 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.2795    1.1321   -0.7210 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.9778   -0.0249   -0.3821 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.4600   -0.0625   -0.5059 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  0
  2  3  1  0
  3  4  2  0
  4  5  1  0
  5  6  2  0
  6  7  1  0
  6  1  1  0
M  END

"""
benzene = oddt.toolkit.readstring('sdf', benzene_sdf)
benzene.calccharges()
methylo = oddt.toolkit.readstring('sdf', methylo_sdf)
methylo.calccharges()


def test_usr():
    """USR test"""
    benzene_usr = np.array((1.392709, 0., 0.,
                            1.732536, 0.877535, -0.580651,
                            1.732576, 0.877555, -0.580693,
                            1.732573, 0.877552, -0.580686))
    methylo_usr = np.array((1.57291, 0.20536, 0.064638,
                            1.697949, 0.759765, -0.420457,
                            2.614764, 1.94016, -1.660038,
                            2.095664, 1.54289, 0.117823))

    assert_array_almost_equal(usr(benzene), benzene_usr)
    assert_array_almost_equal(usr(methylo), methylo_usr)


def test_usr_cat():
    """USRCAT test"""
    benzene_usrcat = np.array((1.392709, 0., 0.,
                               1.732536, 0.877535, -0.580651,
                               1.732576, 0.877555, -0.580693,
                               1.732573, 0.877552, -0.580686,
                               1.392709, 0., 0.,
                               1.732536, 0.877535, -0.580651,
                               1.732576, 0.877555, -0.580693,
                               1.732573, 0.877552, -0.580686,
                               1.392709, 0., 0.,
                               1.732536, 0.877535, -0.580651,
                               1.732576, 0.877555, -0.580693,
                               1.732573, 0.877552, -0.580686,
                               0., 0., 0.,
                               0., 0., 0.,
                               0., 0., 0.,
                               0., 0., 0.,
                               0., 0., 0.,
                               0., 0., 0.,
                               0., 0., 0.,
                               0., 0., 0.))
    methylo_usrcat = np.array((1.57291, 0.20536, 0.064638,
                               1.697949, 0.759765, -0.420457,
                               2.614764, 1.94016, -1.660038,
                               2.095664, 1.54289, 0.117823,
                               1.57291, 0.20536, 0.064638,
                               1.697949, 0.759765, -0.420457,
                               2.614764, 1.94016, -1.660038,
                               2.095664, 1.54289, 0.117823,
                               1.423502, 0.083326, -0.003973,
                               1.732968, 0.877809, -0.58125,
                               3.050558, 0.934105, -0.26119,
                               1.732674, 0.877707, -0.58073,
                               0., 0., 0.,
                               0., 0., 0.,
                               0., 0., 0.,
                               0., 0., 0.,
                               0., 0., 0.,
                               0., 0., 0.,
                               0., 0., 0.,
                               0., 0., 0.))

    assert_array_almost_equal(usr_cat(benzene), benzene_usrcat)
    assert_array_almost_equal(usr_cat(methylo), methylo_usrcat)


def test_electroshape():
    """Electroshape test"""
    if oddt.toolkit.backend == 'rdk':
        benzene_usr = np.array((1.392709, 0.000024, 0.000021,
                                1.732576, 0.936779, -0.834287,
                                1.732573, 0.936777, -0.834284,
                                1.482042, 0.477624, -0.293458,
                                1.482042, 0.477624, -0.293458))
        methylo_usr = np.array((1.58227, 0.461717, 0.417984,
                                2.649053, 1.404615, -1.20629,
                                2.103215, 1.252546, 0.544395,
                                2.009275, 0.651085, -0.488366,
                                1.955503, 0.693917, -0.521708))
    else:
        benzene_usr = np.array((1.39270878, 0.000023678, 0.0000208289,
                                1.73257649, 0.936778843, -0.834286869,
                                1.73257256, 0.936777234, -0.834283948,
                                1.48204167, 0.477623908, -0.293458485,
                                1.48204167, 0.477623908, -0.293458485))
        methylo_usr = np.array((1.61146379, 0.46305567, 0.46938854,
                                2.76274943, 1.37034833, -1.2889241,
                                2.11388564, 1.2641263, 0.57394201,
                                2.18083588, 0.57593416, -0.48947371,
                                2.06548012, 0.71989938, -0.44118673))

    assert_array_almost_equal(electroshape(benzene), benzene_usr)
    assert_array_almost_equal(electroshape(methylo), methylo_usr)


def test_usr_similarity():
    """Similarity function for USR test"""
    similarity_usr = 0.68517293875665453
    assert_almost_equal(usr_similarity(usr(benzene), usr(methylo)), similarity_usr)


def test_usrcat_similarity():
    """Similarity function for USRCAT test"""
    similarity_usrcat = 0.82370755989115707
    assert_almost_equal(usr_similarity(usr_cat(benzene), usr_cat(methylo)),
                        similarity_usrcat)


def test_elecroshape_similarity():
    """Similarity function for Electroshape test"""
    if oddt.toolkit.backend == 'rdk':
        similarity_electroshape = 0.69110953340712389
    else:
        similarity_electroshape = 0.67710967360721197

    assert_almost_equal(usr_similarity(electroshape(benzene), electroshape(methylo)),
                        similarity_electroshape)
