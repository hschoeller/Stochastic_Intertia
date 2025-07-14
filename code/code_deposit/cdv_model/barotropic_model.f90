!Joshua Dorrington 22/03/18 - University of Oxford
!This is the main executable program.
!
!This model solves a 6d spectral truncation of a barotropic beta plane model as
! first presented in [Charney & DeVore 1979]

program barotropic_model
   use, intrinsic :: iso_fortran_env, only: real64
   use params
   use coeffs, only: get_coeffs, build_lin_op
   use utils, only: time_seed, generate_random_ic
   use barotropic6d, only: run_model
   implicit none

   ! Local variables
   real(dp), dimension(dims)             :: init_con
   real(dp), dimension(sample_num,dims) :: state_vector
   real(dp), dimension(sample_num,dims) :: ftle_hist
   real(dp), dimension(coeff_num)       :: coeff
   real(dp), dimension(dims,dims)       :: lin_op
   integer                               :: inner_loop_size
   integer                               :: i

   ! Variables for runtime argument parsing
   integer          :: len0, arglen, idx, g100
   character(len=32):: progname, arg
   character(len=16):: buf1, buf2

   ! Compute derived loop sizes
   inner_loop_size = step_num / sample_num

   ! Seed random numbers and generate initial condition
   call time_seed()
   init_con = generate_random_ic()
   print *, 'Generated random IC:'
   do i = 1, dims
      print *, 'x(', i, ') = ', init_con(i)
   end do

   Read program name and index argument
   call get_command_argument(0, progname, len0)
   call get_command_argument(1, arg,     arglen)
   if (arglen <= 0) then
      write(*,*) 'Usage: ', trim(progname), ' <index>'
      stop 1
   end if
   read(arg(1:arglen), *) idx

   ! Build filename based on g and index
   g100 = int(g * 100.0_dp + 0.5_dp)
   write(buf1, '(I0)') g100
   write(buf2, '(I0)') idx
   save_file      = 'dataOro' // trim(buf1) // '_' // trim(buf2) // '.bin'
   save_file_ftle = 'ftle_'   // trim(buf1) // '_' // trim(buf2) // '.bin'
   ! save_file      = 'dataOro20.bin'
   ! Generate coefficients and the linear operator
   call get_coeffs(b, beta, g, coeff)
   call build_lin_op(lin_op, coeff)

   ! Run the simulation
   call run_model(init_con, sample_num, state_vector, ftle_hist, coeff, lin_op, inner_loop_size)

   ! Write data to files
   open(10, file=trim(save_file),      access='stream', status='replace')
   write(10) state_vector
   close(10)

   open(11, file=trim(save_file_ftle), access='stream', status='replace')
   write(11) ftle_hist
   close(11)

end program barotropic_model
