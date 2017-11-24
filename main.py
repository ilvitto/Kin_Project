import Kin


kin = Kin.Kin()
kin.load("./dataset/0058/body_and_face.mat")

for i in range(len(kin._frames)):
    if kin._frames[i]._body is not None:
        print kin._frames[i]._body._leftHandState