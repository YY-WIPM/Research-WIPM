"""
Created on %(date)s
@author: Y1Y0U	
"""
import numpy as np
from numpy import arange
from scipy.linalg import expm


class Spin(object):
	"""
	
	Collect all the physical properties of spin.
	Give spin operators as output
	
	Attributes:
		spinGyroRatio: the gyromagnetic ratio of this spin. in Hz.T^-1.
		spinLabel: the label of this spin, like 1H, 13C, 17O, etc. 
		spinQuantNbr: the spin quantum number of this spin, like, spinQuantNbr of 1H is 0.5, for 2H it is 1, and for 23Na the spinQuantNbr = 1.5. Dimensionless
		isotopeNbr: the isotope number of this spin. for electron, this number equals 0
		spinIsoCS: the isotropic chemical shift. In ppm
		spinAnisoCS: the anisotropic chemical shift. In ppm
		spinEtaCSA: the eta factor that describes the CSA tensor. Dimensionless
		spinQCC: the quadrupolar coupling constant. In Hz
		spinEtaQCC: the eta factor that describes the electric quadrupolar interaction
		
	Returns:
			I: The Identity operator
			z: The Iz operator of single spin
			p: The I+ operator of single spin
			m: The I- operator of single spin
			x: The Ix operator of single spin
			y: The Iy operator of single spin
	Raises:        
		
	"""

	def __init__(self, **kwarg):
		# The constructor, taking in the physical properties
		for i in kwarg:
			self.gyromagneticRatio = kwarg['spinGyroRatio']
			self.spinLabel = kwarg['spinLabel']
			self.isotopeNbr = kwarg['isotopeNbr']
			self.spinQuantNbr = kwarg['spinQuantNbr']
			self.CSiso = kwarg['CSiso']
		
		self.matrixRank = int(2 * self.spinQuantNbr + 1)
		self.zeemanEigenstates = arange(-self.spinQuantNbr, self.spinQuantNbr + 1, 1)
		self.totAngularNbr = self.spinQuantNbr * (self.spinQuantNbr + 1)
		self.I = np.identity(self.matrixRank)
		self.z = np.diag(self.zeemanEigenstates)
		self.p = np.diag(map(lambda x: np.sqrt(self.totAngularNbr - x * (x + 1)), self.zeemanEigenstates)[:-1], +1)
		self.m = np.diag(map(lambda x: np.sqrt(self.totAngularNbr - x * (x - 1)), self.zeemanEigenstates)[1:], -1)
		self.x = self.p + self.m
		self.y = (self.p-self.m)/(0+2j)


N = Spin(spinLabel='H',
		  spinGyroRatio=42.6e6,
		  spinQuantNbr=0.5,
		  isotopeNbr=1,
		  CSiso=1
		  )

E = Spin(spinLabel='e-',
		  spinGyroRatio=42.6e6,
		  spinQuantNbr=0.5,
		  isotopeNbr=0,
		  CSiso=0
		  )


