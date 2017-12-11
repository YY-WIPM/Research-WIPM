import os
import re
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt





class Proc2dDataset(object):
	"""
	
	"""


	def __init__(self, dataset_dir_path, exp_no, proc_no):
		self.dataset_dir_path = dataset_dir_path
		self.exp_no = exp_no
		self.exp_dir_path = dataset_dir_path + str(exp_no)
		self.proc_no = proc_no
		self.proc_dir_path = self.exp_dir_path + '\\' + 'pdata' + '\\' + str(proc_no) 


		self.param_list = ['$SW', '$SW_h', '$O1', '$SFO1', "$OFFSET", '$SI' ,'$SF', '$TD']
		self.param_dict = {}.fromkeys(self.param_list)

		for item in self.param_list:
			temp_acqu = self.getProcParam(param_name=item, file_name='acqu')
			temp_proc = self.getProcParam(param_name=item, file_name='proc')
			self.param_dict[item] = temp_acqu if temp_acqu else temp_proc
		
		self.param_dict['$TD1$'] = self.getProcParam(param_name='$TD', file_name='acqu2')
		
		"""
		self.real_spectrum = np.asarray(self.getSpectrum('r'), dtype=np.float64)
		self.spectrum_axis = np.linspace(self.param_dict['$OFFSET'], self.param_dict['$OFFSET']-self.param_dict['$SW'], self.param_dict['$SI'], endpoint=True, dtype=np.float64)
		
		self.cplx_fid = self.reshapeFid(self.getFid())
		self.td_axis = np.linspace(0,self.param_dict['$TD']/(2*self.param_dict['$SW_h']),self.param_dict['$TD']/2, endpoint=True, dtype=np.float64)
		"""
		
	def getProcParam(self, param_name, file_name):
		if file_name == 'acqu':
			os.chdir(self.exp_dir_path)
			try:
				with open(file_name) as fstream:
					temp = fstream.readlines()
					for line in temp:
						if re.search( re.escape(param_name), line, flags=0):
							return float(line.split()[-1])			
			except IOError:
				print(self.exp_dir_path + file_name + 'does not exist')

		elif file_name == 'acqu2':
			os.chdir(self.exp_dir_path)
			try:
				with open(file_name) as fstream:
					temp = fstream.readlines()
					for line in temp:
						if re.search( re.escape(param_name), line, flags=0):
							return float(line.split()[-1])			
			except IOError:
				print(self.exp_dir_path + file_name + 'does not exist')


		elif file_name == 'proc':
			os.chdir(self.proc_dir_path)
			try:
				with open(file_name) as fstream:
					temp = fstream.readlines()
					for line in temp:
						if re.search( re.escape(param_name), line, flags=0):
							return float(line.split()[-1])
			except IOError:
				print(self.proc_dir_path + file_name + 'does not exist')


	def getSpectrum(self, spectrum_type='r'):
		spectrum_file = '1'+spectrum_type

		try:
			os.chdir(self.proc_dir_path)
			return np.fromfile(spectrum_file, dtype=np.int32)
		except IOError:
			print(self.proc_dir_path + spectrum_file + 'does not exist')

	def plotSpectrum(self):
		plt.plot(self.spectrum_axis, self.real_spectrum)
		plt.gca().invert_xaxis()
		plt.show()

	def getFid(self):
		os.chdir(self.exp_dir_path)
		try:
			temp = np.fromfile('fid', dtype=np.int32)
			
		except IOError:
			print('FID file' + 'does not exist')
		return temp
	
	def reshapeFid(self, raw_fid):
		fid = np.concatenate((raw_fid[138:],np.zeros(138)),axis=0)
		cmplx_fid=fid[0::2]+1j*fid[1::2]
		return cmplx_fid
	
	def plotFid(self, part='r'):
		if part == 'i':
			plt.plot(self.td_axis,self.cplx_fid.imag)
		else:
			plt.plot(self.td_axis,self.cplx_fid.real)
		plt.show()
		
	def getSer(self):
		os.chdir(self.exp_dir_path)
		try:
			temp = np.fromfile('ser', dtype=np.int32)
			
		except IOError:
			print('SER file' + 'does not exist')
		return temp
	
	
	
	
	def integrateRegion(self):
		return None


if __name__ == "__main__":

	dataset_dir_path='C:/Bruker/TopSpin3.5.b.91pl7/examdata/NMR Data/xj-20171117/'
	exp_no = 1211  
	proc_no = 1
	test_dataset = Proc2dDataset(dataset_dir_path, exp_no, proc_no)
	a = test_dataset.getSer()
	plt.plot(a)
	plt.show()

	
	