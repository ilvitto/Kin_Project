import Kin
import sys


dataset1 = "0064"
dataset2 = "0068"
dataset3 = "0070"
dataset4 = "0065"

#70 - 68 - 64
if len(sys.argv) > 1:
    kin1 = Kin.Kin(dataset_folder=sys.argv[1])
    dataset_number = sys.argv[2]
    if len(sys.argv) > 3 and sys.argv[3] == "clear":
        desc1, classification1 = kin1.run(dataset_number, clear=True)
    else:
        desc1, classification1 = kin1.run(dataset_number)
else:
    print "Choose a dataset!"