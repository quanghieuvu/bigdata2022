import sys
import runs
from runs import *


if __name__ == '__main__':
	arg = sys.argv
	runs.__getattribute__(arg[1]).__getattribute__(arg[2])(arch_id=arg[3], model_id=arg[4], task_name=arg[5])