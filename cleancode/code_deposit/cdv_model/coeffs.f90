!Joshua Dorrington 22/03/18 - University of Oxford
!This module generates the coefficients of the barotropic model,
!and also generates the linear time evolution operator

module coeffs
    use, intrinsic :: iso_fortran_env
    use params
    implicit none
    private
    public get_coeffs, build_lin_op


    contains

	    subroutine get_coeffs(b,beta,g,coeff_arr)
		real(dp) :: coeff_arr(11)
		real(dp), intent(in) :: b, beta, g
		coeff_arr(1)=a_coeff(1,b)
		coeff_arr(2)=a_coeff(2,b)
		coeff_arr(3)=beta_coeff(1,b,beta)
		coeff_arr(4)=beta_coeff(2,b,beta)
		coeff_arr(5)=gamma_coeff(1,b,g)
		coeff_arr(6)=gamma_coeff(2,b,g)
		coeff_arr(7)=gamma_prime_coeff(1,b,g)
		coeff_arr(8)=gamma_prime_coeff(2,b,g)
		coeff_arr(9)=d_coeff(1,b)
		coeff_arr(10)=d_coeff(2,b)
		coeff_arr(11)=16._dp*sqrt(2._dp)/(5._dp*pi)

	    end subroutine get_coeffs

            subroutine build_lin_op(mat,coeff)
	        real(dp), dimension(coeff_num),intent(in) :: coeff
		real(dp), dimension(dims,dims) :: mat
		integer :: i, j

		do j=1, dims
		  do i=1, dims
			mat(i,j)=0._dp
		  end do
		end do
		
		do i=1, dims
		  mat(i,i)=-C
		end do
		mat(1,3)=coeff(7)
		mat(2,3)=coeff(3)
		mat(3,1)=-coeff(5)
		mat(3,2)=-coeff(3)
		mat(4,6)=coeff(8)
		mat(5,6)=coeff(4)
		mat(6,4)=-coeff(6)
		mat(6,5)=-coeff(4)
	   	

	    end subroutine build_lin_op

	    function a_coeff(m,b) result(a)
	    	integer, intent(in) :: m
		real(dp), intent(in) :: b
	    	real(dp) :: a
	    	a=8._dp*sqrt(2._dp)*(m**2)*(b**2+(m**2)-1)/(pi*(-1 +4*(m**2))*(b**2+m**2))
	    end function a_coeff

	    function beta_coeff(m,b,beta) result(output)
	    	integer, intent(in) :: m
		real(dp), intent(in) :: beta, b
	    	real(dp) :: output
	    	output=beta*(b**2)/(b**2 + m**2)
	    end function beta_coeff

	    function gamma_coeff(m,b,g) result(gamma_)
	    	integer, intent(in) :: m
		real(dp), intent(in) :: g, b
	    	real(dp) :: gamma_
	    	gamma_=(g*4._dp*sqrt(2._dp)*b*m**3)/((4*(m**2)-1)*pi*(b**2+m**2))
	    end function gamma_coeff
	  
	    function gamma_prime_coeff(m,b,g) result(gamma_p)
		integer, intent(in) :: m
		real(dp), intent(in) :: g,b
		real(dp) :: gamma_p
		gamma_p=(g*4._dp*sqrt(2._dp)*b*m)/((4*(m**2)-1)*pi)
	    end function gamma_prime_coeff

	   function d_coeff(m,b) result(d)
		integer, intent(in) :: m
		real(dp), intent(in) :: b
		real(dp) :: d
		d=(64._dp*sqrt(2._dp)*(b**2-(m**2) + 1))/(15._dp*pi*(b**2 + m**2))
	    end function d_coeff

	   
end module coeffs
