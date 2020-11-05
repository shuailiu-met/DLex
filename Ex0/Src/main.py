from pattern import Checker
from pattern import Circle
from pattern import Spectrum

checker = Checker(512,32)
checker.draw()
checker.show()

circle = Circle(512,125,(333,256))
circle.draw()
circle.show()

spec = Spectrum(256)
spec.draw()
spec.show()