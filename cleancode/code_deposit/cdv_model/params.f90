
!Joshua Dorrington 22/03/18 - University of Oxford
!This module contains all the parameters, physical and otherwise
module params
	
	use, intrinsic :: iso_fortran_env

    	implicit none
	!global constants
	integer, parameter :: dp = real64
	real(dp), parameter :: pi = 4._dp*ATAN(1._dp)
	
	!model timestepping params
	real(dp), parameter :: dt =2e-4_dp
	real(dp), parameter :: t_0=0._dp
	real(dp), parameter :: t_f=200000._dp
	real(dp), parameter :: sample_t =1._dp

		!derived timestepping parameters, don't touch directly
		integer(kind=int64), parameter :: step_num=(t_f-t_0)/dt
		integer, parameter :: sample_num=int((t_f-t_0)/sample_t)
	
	!model params:
	character(len=1024), parameter :: save_file = SUB_SAVE_FILE
	!strength of stochasticity
	real(dp), parameter :: sigma = SUB_SIGMA
	!noise type, "w"=white noise, "r" = red noise
	character, parameter :: noise_type=SUB_NOISE_TYPE
	!correlation of noise (only for noise type ='r')
	real(dp), parameter :: r=SUB_R
	!number of dimensions, and codimensions
	integer, parameter :: dims=6
	integer, parameter :: coeff_num =11
	
	!array like params, the initial condition and the forcing vector
	real(dp), parameter, dimension(dims) :: &
	init_con =(/0.927796_dp,0.160574_dp,-0.0969953_dp,-0.672758_dp,-0.118184_dp,0.214505_dp/)
	real(dp), parameter, dimension(dims) :: &
	xf=(/0.95_dp,0._dp,0._dp,-0.76095_dp,0._dp,0._dp/)

	!forcing magnitude
	real(dp), parameter :: C=0.1_dp
	!strength of coriolis terms
	real(dp), parameter :: beta=1.25_dp
	!geometry of box; ratio of channel length to width	
	real(dp), parameter :: b=0.5_dp
	!topographical height parameter
	real(dp), parameter :: g=0.2_dp
	
	

end module params
