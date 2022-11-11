
from gradvi.utils.unittest_tester import UnittestTester
from gradvi import tests as gvtests

import inspect

def get_test_classes(class_names = []):
    test_classes = []
    for name, obj in inspect.getmembers(gvtests):
        if inspect.isclass(obj):
            if len(class_names) == 0:
                test_classes.append(obj)
            elif name in class_names:
                test_classes.append(obj)
    return test_classes
    

def run_unittests(test_class_names = []):
    test_classes = get_test_classes(test_class_names)
    tester = UnittestTester(test_classes)
    tester.execute()
    del tester
    # =========================
    # if you want report for each class separately,
    # =========================
    #for mtest in test_classes:
    #    tester = UnittestTester(mtest)
    #    del tester
    return
