from pattern import Checker
from pattern import Circle
from pattern import Spectrum
import generator
import os

checker = Checker(512,32)
checker.draw()
checker.show()

circle = Circle(512,125,(333,256))
circle.draw()
circle.show()

spec = Spectrum(256)
spec.draw()
spec.show()

dir_path = 'E:/Deep Learning/Exercise/ex0/src/'
json_path = 'E:/Deep Learning/Exercise/ex0/src/Labels.json'
file_path = os.path.join(dir_path, 'exercise_data/')

gen = generator.ImageGenerator(file_path, json_path, 16, [32, 32, 3], shuffle=False, mirroring=False,rotation=False) #according to the numpytest, image is 32x32 pixels with 3 channels
gen.show()