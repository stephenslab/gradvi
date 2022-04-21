
from gradvi.utils.unittest_tester import UnittestTester

from gradvi.tests import TestNormalMeansPy, \
                         TestLinearModel, \
                         TestNMOperator, \
                         TestOldPLRAsh, \
                         TestNMFromPosterior

def run_unittests():
    for mtest in [TestNormalMeansPy, 
                  TestNMOperator,
                  TestLinearModel,
                  TestOldPLRAsh,
                  TestNMFromPosterior
                 ]:  
        tester = UnittestTester(mtest)
        tester.execute()
        del tester
    return
