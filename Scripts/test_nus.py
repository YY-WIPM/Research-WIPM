
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet

def calLeastSqure(Mat,y):
  temp = np.matmul(Mat.T, Mat)
  temp = np.linalg.pinv(temp)
  temp = temp * Mat.T
  temp = np.matmul(temp, y)
  return temp

class Dataset1D(object):
  """
  Read raw 1D FID in BRUKER convention and reform into an array of time dependent complex numbers
  FID plotting is possible
  
  
  dataset_dir_path='C:/Bruker/TopSpin3.5.b.91pl7/examdata/NMR Data/XJ_LLS_20171130/'
  exp_no = 201 
  proc_no = 1
# dataset_dir_path='C:/NMR Spectra/156574_0001csol/'
# exp_no = 10 
# proc_no = 1
  
  """

  def __init__(self, dataset_dir_path, exp_no, proc_no,pos_shift=138):
    '''
    Requires the absolute directory path of 1D experiment, number of experiment.
    The number of processing is also needed, provided that the processed data is required
    '''

    self.ds_dir_path = dataset_dir_path
    self.exp_no = exp_no
    self.proc_no = proc_no
    self.cmplx_fid = self.mkCplxFid(pos_shift)
    self.proc_spec = self.getProcSpec()

  def getFid(self):
    try:
      fid_dir_path = self.ds_dir_path + str(self.exp_no)
      os.chdir(fid_dir_path)
      temp = np.fromfile('fid', dtype=np.int32)
    except IOError:
      print('FID file does not exist')
    return temp

  def mkCplxFid(self,pos_shift):
    temp = self.getFid()
    temp = np.concatenate((temp[pos_shift:], np.zeros(pos_shift)), axis=0)
    cmplx_fid = temp[0::2] + 1j * temp[1::2]
    return cmplx_fid

  def plotFid(self, part='r',pos_shift=138):
    '''
    plot either the real part ('r') or the imaginary part ('i') of the complex FID
    '''

    if part == 'r':
      plt.plot(self.cmplx_fid.real)
    elif part == 'i':
      plt.plot(self.cmplx_fid.imag)
    plt.show()

  def plotFT(self):
    temp = np.fft.fft(self.cmplx_fid)
    temp = np.fft.fftshift(temp)
    plt.plot(temp)
    plt.show()

 

  def getProcSpec(self):
    try:
      proc_dir_path = self.ds_dir_path + str(self.exp_no) + '/pdata/' + str(self.proc_no)
      os.chdir(proc_dir_path)
      temp = np.fromfile('1r', dtype=np.int32)
    except IOError:
      print('Processed spectrum does not exist')
    return temp         



  def plotProcSpec(self):
    temp = self.getProcSpec()
    plt.plot(temp)
    plt.show()

class RandCompSensing(object):
  """
  Sparsity Adaptive Matching Persuit
  According to the theory of Compressed Sensing, we have:
  
    y_(Nx1) = Phi(NxM) * x(Mx1)
    y : N-entry-column-vector of measured data
    Phi : NxM-matrix of measuring matrix
    x : M-entry-column-vector of original data
  First we construct a random sensing matrix(RSM), where N equals to the length of complex FID from experimental dataset. M, the length of the array of measurement, has a theoretical lower limit, which can be obtained from the Theorem of Measurement Bound:
    M >= C*k*log(n/k),
    C=\frac{1}{2(1+\sqrt{24})} = 0.28,
    k = nbr of non-zero freqs
  We know already the value of k in a deterministic way. So we first set k = 2, then m > 0.28*2*4.21 = 2.35.
  In practice, we adopt in the first attempts M = 4k, sparsity 12.5%
  
  Then the task is translated into a reconstruction problem: with known Gaussian random sensing matrix and Gaussian measured data, we reconstruct the original processed spectrum
  """

  def __init__(self, original_dataset, sparsity=0.125):
    self.original_sig = original_dataset.proc_spec
    self.len_original = len(original_dataset.proc_spec)
    self.sparsity = sparsity
    self.len_measure = np.floor(self.len_original*self.sparsity)
    self.gaussian_RSM = self.mkGaussianRSM()
    self.g_measured_sig = np.dot(self.gaussian_RSM,original_dataset.proc_spec)

  def mkGaussianRSM(self):
    temp = np.random.normal(0, 1.0, int(self.len_measure*self.len_original))
    temp = np.reshape(temp, (-1, self.len_original))
    return temp

  def mkBernoulliRSM(self):
    temp = np.random.binomial(1,0.5, self.len_measure*self.len_original)
    temp = np.reshape(temp, (-1, self.len_original))
    return temp
  
class NormalSignal(object):
  def __init__(self, sig_len,sparsity):
    self.sig_len = sig_len
    self.proc_spec = self.mkNormalSignal(self.sig_len,sparsity)
    plt.plot(self.proc_spec)
    plt.show()
    
    
  def mkNormalSignal(self, sig_len,sparsity):
    temp = np.zeros(sig_len)
    temp2 = np.random.randint(int(sig_len),size=int(sig_len*sparsity))
    temp[[temp2]]=np.random.normal(0,1,int(sig_len*sparsity))
    return temp

class PseudoSignal(object):
  """
  For the purpose of algorithm test, we avoid real dense spectrum, and we first try with ideal binomial distribution as pseudo-signal generator.
  """
  def __init__(self, nbr_pts, sparsity):
    self.nbr_pts = nbr_pts
    self.sparsity = sparsity
    self.proc_spec = self.mkPseudoSignal(self.nbr_pts, self.sparsity)
    plt.plot(self.proc_spec)
    plt.show()
    
  def mkPseudoSignal(self, nbr_pts, sparsity):
    l0_norm = int(np.floor(sparsity*nbr_pts))
    sig = np.random.binomial(1, sparsity, nbr_pts)
    
    while sum(sig)!=l0_norm:
      sig = np.random.binomial(1, sparsity, nbr_pts)
      
    print(sum(sig))
    return sig

class OMP(object):
  """
  y(Nx1): the signal from compressed sensing
  Mat(MxN): the matrix of compressed sensing
  K: number of non-zero entries in the original signal (sadly it is somehow deterministic knowledge...).
  
  reconstruct_sig(Mx1): the signal reconstructed from l_1 optimisation 
  residual: residual..
  
  """
  def __init__(self, y, Mat, K):

    self.reconstruct_sig, self.residual = self.optL1(y, Mat, K)

  def optL1(self, y, Mat, K):
    iteration = K
    M,N = np.shape(Mat)
    y = np.array([y]).T
    theta = np.zeros(N)
    A = Mat
    At = np.zeros((M,iteration))
    pos_num = np.zeros(iteration)
    res = y
    for iter in range(iteration):
      # Calculate the inner products between measured/compressed signal with all columns in the sensing matrix
      product = np.dot(A.T,res)
      # Locate the max abs value among the inner products
      abs_product = np.abs(product)
      pos = np.argpartition(abs_product.flatten(),-1)[-1:]
      # Update the index of mas abs inner product and the set of column vectors that gives max abs inner products.
      pos_num[iter] = pos

      # Set the column of max abs inner product to zero 
      for ii in range(M):
        At[ii,iter] = A[ii,pos]
        A[ii,pos] = 0
      
      # Calculate the least square fit of y=At.theta_t
      theta_ls = self.calLeastSquare(At[:,:iter+1],y)
      res = y - np.dot(At[:,:iter+1], theta_ls)
      
      # print("val=",val_num,'\n', "pos=",pos_num,'\n',"theta_ls=",theta_ls)
      # res = y-np.dot(At[:,iter:iter+1], theta_ls)
      
    theta[[int(i) for i in pos_num]]=theta_ls.flatten()
    return theta, np.linalg.norm(res)

  
  def calLeastSquare(self, Mat, y):
    """
    Calculate the least square solution theta_ls of the optimisation problem:
      Mat (\cdot) theta = y
    """
    temp1 = np.matmul(Mat.T, Mat)
    temp2 = np.matmul(Mat.T, y)
    temp = np.linalg.solve(temp1, temp2)
    return temp
    

class SAMP(object):
  """
  y(Nx1): the signal from compressed sensing
  Mat(MxN): the matrix of compressed sensing
  S: steplength of increment
  
  reconstruct_sig(Mx1): the signal reconstructed from l_1 optimisation 
  residual: residual..
  """
  def __init__(self, y, Mat, S=1):
    self.reconstruct_sig, self.residual = self.optL1(y, Mat, K)
    
  def optL1(self, y, Mat, K):
    M,N = np.shape(Mat)
    y = np.array([y]).T
    theta = np.zeros(N)
    A = Mat
    At = np.zeros((M,M*S))
    pos_num = np.zeros(M*S)
    res = y
    L = S
    for t in range(N):
      abs_product = np.abs(np.matmul(A.T, res))
      s_k = np.argpartition(abs_product.flatten(),-1)[-1:]
      pos_num[t] = s_k
      At[:,t] = A[:,s_k].flatten()
      A[:,s_k] = np.array([np.zeros(M)]).T
      theta_ls = calLeastSquare(At[:,:L],y)
      res_new = y - np.dot(At[:,:L], theta_ls)
      if np.linalg.norm(res_new) > np.linalg.norm(res): 
          print(L)
          break
      else: 
          res= res_new
          L = L+S
    for i in range(len(theta_ls)):
      theta[int(pos_num[i])]=theta_ls.flatten()[i]
    return theta, np.linalg.norm(res_new)





      
      



if __name__ == "__main__":
  dataset_dir_path = 'C:/Bruker/TopSpin3.5.b.91pl7/examdata/NMR Data/XJ_LLS_20171130/'
  # dataset_dir_path = 'C:/NMR Spectra/156574_0001csol/'
  exp_no = 10 # 201
  proc_no = 1
  test_dataset = Dataset1D(dataset_dir_path, exp_no, proc_no,pos_shift=125)
  # test_dataset.plotFid()
  # test_dataset.plotProcSpec()

  """



  plt.plot(test_dataset.proc_spec)

  for K in [16,64,128,144,160]:
  # for K in [1,8,16,64,128,192,256,320,384,512,800,1024]:
  
    test_RS = RandCompSensing(test_dataset, sparsity=M/N)
    
    foo = OMP(test_RS.g_measured_sig, test_RS.gaussian_RSM,K)
  
    a = foo.reconstruct_sig#
    b = test_RS.original_sig
    plt.plot(b-a)
    res = foo.residual
    print(K, np.linalg.norm(a-b), res )
    
  plt.show()
  """

  '''
  def calLeastSquare( Mat, y):
    """
    Calculate the least square solution theta_ls of the optimisation problem:
      Mat (\cdot) theta = y
    """
    temp1 = np.matmul(Mat.T, Mat)
    temp2 = np.matmul(Mat.T, y)
    temp = np.linalg.solve(temp1, temp2)
    return temp
  '''

  def calLeastSquare(Mat, y, alpha=.1):
    reg = ElasticNet(alpha,l1_ratio=1)
    temp = np.asarray([reg.fit(Mat,y).coef_]).T
    return temp

  N = len(test_dataset.proc_spec) # length of real signal
  M = N*0.35# length of measured signal
  S = 1
  test_RS = RandCompSensing(test_dataset, sparsity=M/N)
  y = test_RS.g_measured_sig
  Mat = test_RS.gaussian_RSM
  M,N = np.shape(Mat)
  y = np.array([y]).T
  theta = np.zeros(N)
  A = Mat
  At = np.zeros((M,M*S))
  pos_num = np.zeros(M*S)
  res = y
  L = S
  
  for K in [M]:
      for t in range(K):
       
        abs_product = np.abs(np.matmul(A.T, res))
        s_k = np.argpartition(abs_product.flatten(),-1)[-1:]
        pos_num[t] = s_k
        At[:,t] = A[:,s_k].flatten()
        A[:,s_k] = np.array([np.zeros(M)]).T
        theta_ls = calLeastSquare(At[:,:L],y)
        res_new = y - np.dot(At[:,:L], theta_ls)
        if np.linalg.norm(res_new) > np.linalg.norm(res): 
            print(L)
            break
        else: 
            res= res_new
            L = L+S
      for i in range(len(theta_ls)):
        theta[int(pos_num[i])]=theta_ls.flatten()[i]
      plt.plot(theta)