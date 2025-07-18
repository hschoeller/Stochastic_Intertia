!Joshua Dorrington 22/03/18 - University of Oxford
!This module handles random number generation for stochastic runs
module utils
   use, intrinsic :: iso_fortran_env
   use params
   implicit none
   private
   public time_seed, white_noise, red_noise, generate_random_ic

contains

!lifted wholesale from Sam Hatfield's Lorenz63-4D Var model
!seeds random numbers
   subroutine time_seed()
      integer :: i, n, clock
      integer, allocatable :: seed(:)
      call random_seed(size = n)
      allocate(seed(n))
      call system_clock(count=clock)
      seed = clock + 37 * (/ (i - 1, i = 1, n) /)
      call random_seed(put = seed)
      deallocate(seed)
   end subroutine

!generates (n X dims) array of gaussian random numbers using Box Muller
   subroutine white_noise(rand_arr,mean,std,n)
      real(dp) :: rand_arr(:,:)
      real(dp), intent(in) :: mean, std
      integer , intent(in):: n
      real(dp) :: rand1(n,dims), rand2(n,dims),u(n,dims),v(n,dims)
      !random univariates
      call random_number(rand1)
      call random_number(rand2)
      !Box-Muller
      u = sqrt(-2._dp * log(rand1))
      v = cos(8._dp*ATAN(1._dp) * rand2)
      rand_arr = mean + (std * u * v)
   end subroutine

!generates (n X dims) array of red noise with lag 1 correlation r
   subroutine red_noise(rand_arr, mean,std,n)
      real(dp) :: rand_arr(:,:), white_arr(n,dims)
      real(dp), intent(in) :: mean, std
      integer , intent(in):: n
      integer :: i
      real(dp) :: k
      k=sqrt(1-r*r)
      call white_noise(white_arr,mean,std,n)
      rand_arr(1,:)=white_arr(1,:)
      do i=1, n-1
         rand_arr(i+1,:)=r*rand_arr(i,:) + k*white_arr(i+1,:)
      end do
   end subroutine

!generates random initial condition within specified ranges
   function generate_random_ic() result(random_ic)
      real(dp) :: random_ic(dims)
      real(dp) :: random_uniform(dims)

      ! Define ranges for each dimension
      real(dp), parameter :: ic_min(dims) = &
         (/0.8_dp, -0.1_dp, -0.4_dp, -0.7_dp, -0.4_dp, -0.2_dp/)
      real(dp), parameter :: ic_max(dims) = &
         (/ 0.95_dp,  0.3_dp,  0.1_dp,  -0.2_dp,  0.4_dp,  0.45_dp/)

      ! Generate uniform random numbers [0,1]
      call random_number(random_uniform)

      ! Scale to desired ranges
      random_ic = ic_min + random_uniform * (ic_max - ic_min)

   end function generate_random_ic

end module utils
