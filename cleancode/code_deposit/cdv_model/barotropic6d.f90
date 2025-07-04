
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


	contains
		subroutine run_model(init_con,stepnum,samplenum,output, coeff,lin_op,n_inner)

			integer(kind=int64), intent(in) :: stepnum
			integer, intent(in) :: samplenum, n_inner
			real(dp), intent(in) :: init_con(dims)
			real(dp), dimension(samplenum, dims) :: output
			real(dp), dimension(dims) :: k1, k2,x
			real(dp),intent(in), dimension(coeff_num) :: coeff
			real(dp), intent(in), dimension(dims,dims) :: lin_op
			integer :: i, j
			real(dp), dimension(n_inner,dims) :: stoch_arr
			real(dp) :: a1,a2,beta1,beta2,gamma1,gamma2,gammaprime1, &
				    gammaprime2, d1, d2, e
			
			!Unpacks the model coefficients
			a1=coeff(1)
			a2=coeff(2)
			beta1=coeff(3)
			beta2=coeff(4)
			gamma1=coeff(5)
			gamma2=coeff(6)
			gammaprime1=coeff(7)
			gammaprime2=coeff(8)
			d1=coeff(9)
			d2=coeff(10)
			e=coeff(11)
			
			! First step
	       		output(1,:) = init_con
			x=init_con
			
			!all other steps, sampling at end of inner loop
			do i = 1, size(output,1)-1
				
				!generate random numbers for next inner loop
				if (noise_type=="w") then
					call white_noise(stoch_arr,0._dp,sqrt(dt),n_inner)
				else if (noise_type=="r") then
					call red_noise(stoch_arr,0._dp,sqrt(dt),n_inner)
				end if

				do j = 1, n_inner
		    			k1 = dt*dxdt(x)
		    			!k2 = dt*dxdt(x + k1)

		    			!x = x + 0.5_dp * (k1 + k2)
					x=x+k1+sigma*stoch_arr(j,:)
				end do
				output(i+1,:)=x
				!print*,(100._dp*(i-1))/size(output,1), " percent complete"
			end do

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
		end subroutine run_model
		
		


end module barotropic6d
