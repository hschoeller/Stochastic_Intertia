!Joshua Dorrington 22/03/18 - University of Oxford
!this module contains the model equations and integrates
! them using a first order Euler Maruyama scheme

module barotropic6d
   use coeffs
   use params
   use utils, only: white_noise, red_noise
   implicit none
   private
   public run_model

   interface
      ! LAPACK thin‐QR: factors A in place (upper triangle = R, lower+τ = reflectors)
      ! subroutine dgeqrf(m, n, A, lda, tau, work, lwork, info) bind(C,name="dgeqrf_")
      !    use, intrinsic :: iso_c_binding
      !    integer(c_int), intent(in)    :: m,n,lda,lwork
      !    real(c_double), intent(inout):: A(lda,*), work(*)
      !    real(c_double), intent(out)  :: tau(min(m,n))
      !    integer, intent(out)   :: info
      ! end subroutine
      ! LAPACK form-Q: builds orthonormal Q from the reflectors in A
      ! subroutine dorgqr(m, n, k, A, lda, tau, work, lwork, info) bind(C,name="dorgqr_")
      !    use, intrinsic :: iso_c_binding
      !    integer, intent(in)    :: m,n,k,lda,lwork
      !    real(c_double), intent(inout):: A(lda,*), work(*)
      !    real(c_double), intent(in)   :: tau(k)
      !    integer, intent(out)   :: info
      ! end subroutine
      ! LAPACK eigenvalue routine for real general matrices
      subroutine dgeev(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info) bind(C,name="dgeev_")
         use, intrinsic :: iso_c_binding
         character(c_char), intent(in) :: jobvl, jobvr
         integer(c_int), intent(in) :: n, lda, ldvl, ldvr, lwork
         real(c_double), intent(inout) :: a(lda,*)
         real(c_double), intent(out) :: wr(*), wi(*), vl(ldvl,*), vr(ldvr,*), work(*)
         integer(c_int), intent(out) :: info
      end subroutine
   end interface

contains
   subroutine run_model(init_con, samplenum, output, ftle_hist, coeff, lin_op, n_inner)
      integer, intent(in)             :: samplenum, n_inner
      real(dp), intent(in)            :: init_con(dims)
      real(dp), intent(in)            :: coeff(coeff_num)
      real(dp), intent(in)            :: lin_op(dims,dims)
      real(dp), dimension(samplenum,dims), intent(out) :: output
      real(dp), dimension(samplenum,dims), intent(out) :: ftle_hist
      real(dp), dimension(dims) :: k1, k2,x
      real(dp), dimension(n_inner,dims) :: stoch_arr
      real(dp) :: a1,a2,beta1,beta2,gamma1,gamma2,gammaprime1, &
         gammaprime2, d1, d2, e

      real(dp) :: Jacob(dims,dims)
      complex(dp) :: eig_vals(dims)
      ! real(dp) :: M(dims,dims), Mdot(dims,dims)
      ! real(dp) :: tau(dims), work_qr(4*dims)
      ! integer  :: info
      ! real(dp) :: sumlog(dims)
      ! real(dp) :: Rkk
      integer  :: i,j,k

      ! Variables for DGEEV eigenvalue calculation
      real(dp) :: wr(dims), wi(dims)          ! Real and imaginary parts of eigenvalues
      real(dp) :: vl(dims,dims), vr(dims,dims) ! Left and right eigenvectors (not used)
      real(dp), allocatable :: work_eig(:)     ! Work array for DGEEV
      real(dp) :: work_query(1)                ! For querying optimal work size
      integer :: lwork_eig                     ! Size of work array
      integer :: info                          ! Moved here from commented section

      !Unpacks the model coefficients
      a1=coeff(1)
      a2=coeff(2)
      ! beta1=coeff(3)
      ! beta2=coeff(4)
      ! gamma1=coeff(5)
      ! gamma2=coeff(6)
      ! gammaprime1=coeff(7)
      ! gammaprime2=coeff(8)
      d1=coeff(9)
      d2=coeff(10)
      e=coeff(11)

      ! Query optimal workspace size for DGEEV
      call dgeev('N', 'N', dims, Jacob, dims, wr, wi, vl, dims, vr, dims, work_query, -1, info)
      if (info /= 0) then
         print *, 'Error in DGEEV workspace query: info = ', info
         stop
      end if
      lwork_eig = int(work_query(1))
      allocate(work_eig(lwork_eig))

      ! First step
      output(1,:)   = init_con
      ftle_hist(1,:)= 0._dp
      x = init_con
      !M = 0._dp
      !do k=1,dims; M(k,k)=1._dp; end do
      !sumlog = 0._dp

      !all other steps, sampling at end of inner loop
      do i = 1, samplenum-1

         !generate random numbers for next inner loop
         if (noise_type=="w") then
            call white_noise(stoch_arr,0._dp,sqrt(dt),n_inner)
         else if (noise_type=="r") then
            call red_noise(stoch_arr,0._dp,sqrt(dt),n_inner)
         end if

         do j = 1, n_inner
            ! use midpoint rule for higher accuracy in fltes
            k1 = dt*dxdt(x)
            k2 = dt * dxdt(x + 0.5_dp * k1)
            x=x+k2+sigma*stoch_arr(j,:)

            ! This is Heuns method:
            !k2 = dt*dxdt(x + k1)
            !x = x + 0.5_dp * (k1 + k2)

            ! variational eqn: Mdot = Jacob * M
            ! Mdot = matmul(build_jacobian(x), M)
            ! M    = M + dt * Mdot
            ! ! thin‐QR of M -> (reflectors in M, diag of R in tau & upper triangle)
            ! call dgeqrf(dims, dims, M, dims, tau, work_qr, size(work_qr), info)

            ! ! extract Rkk from the (k,k) entry of the upper triangle
            ! do k=1,dims
            !    Rkk = M(k,k)
            !    sumlog(k) = sumlog(k) + log(abs(Rkk))
            ! end do

            ! ! form Q in place of M
            ! call dorgqr(dims, dims, dims, M, dims, tau, work_qr, size(work_qr), info)

         end do
         output(i+1,:)=x

         ! Compute eigenvalues of the Jacobian at current state
         Jacob = build_jacobian(x)

         ! Compute eigenvalues using DGEEV
         call dgeev('N', 'N', dims, Jacob, dims, wr, wi, vl, dims, vr, dims, work_eig, lwork_eig, info)

         if (info /= 0) then
            print *, 'Error in DGEEV eigenvalue calculation: info = ', info
            print *, 'At iteration i = ', i
            stop
         end if

         ! Convert to complex eigenvalues and store real parts in ftle_hist
         do k=1,dims
            eig_vals(k) = cmplx(wr(k), wi(k), kind=dp)
            ftle_hist(i+1,k) = real(eig_vals(k))  ! Store real part of eigenvalue
            ! ftle_hist(i+1,k) = sumlog(k) / ( (i*n_inner) * dt )
         end do

         print*,(100._dp*(i-1))/size(output,1), " percent complete"
      end do

      ! Clean up allocated memory
      deallocate(work_eig)

   contains

      !The o.d.e is here
      function dxdt(x)

         real(dp),intent(in) :: x(dims)
         real(dp) :: dxdt(dims)

         !applies linear matrix operator
         dxdt=matmul(lin_op,x)+C*xf
         !adds non linear terms
         dxdt=dxdt+(/0._dp,&
            -a1*x(1)*x(3)-d1*x(4)*x(6),&
            a1*x(1)*x(2)+d1*x(4)*x(5),&
            e*(x(2)*x(6)-x(3)*x(5)), &
            -a2*x(1)*x(6)-d2*x(3)*x(4), &
            a2*x(1)*x(5)+d2*x(2)*x(4)/)

      end function dxdt

      function build_jacobian(x)
         real(dp), intent(in) :: x(dims)
         real(dp) :: build_jacobian(dims,dims)

         ! Initialize Jacobian with the linear operator
         build_jacobian = lin_op

         ! Add nonlinear terms' partial derivatives
         ! Row 1: d/dx of component 1 (no nonlinear terms, just linear)

         ! Row 2: d/dx of component 2 = -a1*x(1)*x(3) - d1*x(4)*x(6)
         build_jacobian(2,1) = build_jacobian(2,1) - a1*x(3)   ! ∂/∂x1
         build_jacobian(2,3) = build_jacobian(2,3) - a1*x(1)   ! ∂/∂x3
         build_jacobian(2,4) = build_jacobian(2,4) - d1*x(6)   ! ∂/∂x4
         build_jacobian(2,6) = build_jacobian(2,6) - d1*x(4)   ! ∂/∂x6

         ! Row 3: d/dx of component 3 = a1*x(1)*x(2) + d1*x(4)*x(5)
         build_jacobian(3,1) = build_jacobian(3,1) + a1*x(2)   ! ∂/∂x1
         build_jacobian(3,2) = build_jacobian(3,2) + a1*x(1)   ! ∂/∂x2
         build_jacobian(3,4) = build_jacobian(3,4) + d1*x(5)   ! ∂/∂x4
         build_jacobian(3,5) = build_jacobian(3,5) + d1*x(4)   ! ∂/∂x5

         ! Row 4: d/dx of component 4 = e*(x(2)*x(6) - x(3)*x(5))
         build_jacobian(4,2) = build_jacobian(4,2) + e*x(6)    ! ∂/∂x2
         build_jacobian(4,3) = build_jacobian(4,3) - e*x(5)    ! ∂/∂x3
         build_jacobian(4,5) = build_jacobian(4,5) - e*x(3)    ! ∂/∂x5
         build_jacobian(4,6) = build_jacobian(4,6) + e*x(2)    ! ∂/∂x6

         ! Row 5: d/dx of component 5 = -a2*x(1)*x(6) - d2*x(3)*x(4)
         build_jacobian(5,1) = build_jacobian(5,1) - a2*x(6)   ! ∂/∂x1
         build_jacobian(5,3) = build_jacobian(5,3) - d2*x(4)   ! ∂/∂x3
         build_jacobian(5,4) = build_jacobian(5,4) - d2*x(3)   ! ∂/∂x4
         build_jacobian(5,6) = build_jacobian(5,6) - a2*x(1)   ! ∂/∂x6

         ! Row 6: d/dx of component 6 = a2*x(1)*x(5) + d2*x(2)*x(4)
         build_jacobian(6,1) = build_jacobian(6,1) + a2*x(5)   ! ∂/∂x1
         build_jacobian(6,2) = build_jacobian(6,2) + d2*x(4)   ! ∂/∂x2
         build_jacobian(6,4) = build_jacobian(6,4) + d2*x(2)   ! ∂/∂x4
         build_jacobian(6,5) = build_jacobian(6,5) + a2*x(1)   ! ∂/∂x5
      end function build_jacobian

   end subroutine run_model

end module barotropic6d
