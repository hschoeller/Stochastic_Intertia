!Joshua Dorrington 22/03/18 - University of Oxford
!This is the main executable program.

!This model solves a 6d spectral truncation of a baratropic beta plane model as
! first presented in [Charney & DeVore 1979] 


!all model parameters are in params
!formulas for all spectral coefficients are in coeffs
!random number generation is in utils
!the model equations and integrator are in baratropic6d

program barotropic_model
	use, intrinsic :: iso_fortran_env
	use params
	use coeffs, only: get_coeffs, build_lin_op
	use utils, only: time_seed
	use barotropic6d, only: run_model
	implicit none
	
	
	!Initialise data structures
	real(dp), dimension(sample_num,dims) :: state_vector
	real(dp), dimension(coeff_num) :: coeff
	real(dp), dimension(dims,dims) :: lin_op
	integer :: inner_loop_size

 
	
	inner_loop_size=step_num/sample_num

	!Seed random numbers
	call time_seed()

	!Generate coefficients and the linear time evolution operator
	call get_coeffs(b,beta,g,coeff)
	call build_lin_op(lin_op,coeff)
	
	!Run the simulation
	call run_model(init_con,step_num,sample_num,state_vector,coeff,lin_op,inner_loop_size)
	
	!Write data to file
	OPEN( 10, FILE=save_file, access="stream")
	WRITE(10) state_vector   
end program
