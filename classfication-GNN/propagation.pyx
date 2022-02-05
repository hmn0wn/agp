from propagation cimport Agp

cdef class AGP:
	cdef Agp c_agp

	def __cinit__(self):
		self.c_agp=Agp()

	def agp_operation(self,dataset,agp_alg,unsigned int m,unsigned int n,int L,rmax,alpha,t,np.ndarray array3, double[:] prep_t, double[:] cclock_t):
		return self.c_agp.agp_operation(dataset.encode(),agp_alg.encode(),m,n,L,rmax,alpha,t,Map[MatrixXd](array3), prep_t[0], cclock_t[0])