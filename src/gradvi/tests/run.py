
from gradvi.utils.unittest_tester import UnittestTester

from gradvi.tests import TestNormalMeansPy, \
                         TestLinearModel, \
                         TestNMOperator, \
                         TestOldPLRAsh

def run_unittests():
    for mtest in [TestNormalMeansPy, 
                  TestNMOperator,
                  TestLinearModel,
                  TestOldPLRAsh,
                 ]:  
        tester = UnittestTester(mtest)
        tester.execute()
        del tester
    return
