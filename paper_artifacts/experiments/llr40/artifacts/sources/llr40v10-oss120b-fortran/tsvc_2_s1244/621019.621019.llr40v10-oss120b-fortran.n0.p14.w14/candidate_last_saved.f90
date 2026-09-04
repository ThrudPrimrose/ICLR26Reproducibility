module tsvc_2_s1244_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s1244_fp64(a, b, c, d, LEN_1D) bind(C, name="tsvc_2_s1244_fp64")
    ! Arguments: a, b, c, d are double precision arrays, LEN_1D is int64
    real(c_double), intent(inout) :: a(0:*)
    real(c_double), intent(in)    :: b(0:*), c(0:*)
    real(c_double), intent(out)   :: d(0:*)
    integer(c_int64_t), value    :: LEN_1D
    integer(c_int64_t) :: i
    real(c_double), allocatable :: a_old(:)

    if (LEN_1D <= 1_c_int64_t) then
      return
    end if

    allocate(a_old(0:LEN_1D-1))
    a_old(0:LEN_1D-1) = a(0:LEN_1D-1)

    !$omp parallel do default(none) shared(a,b,c,d,a_old,LEN_1D) private(i)
    do i = 0, LEN_1D - 2
      a(i) = b(i) + c(i)*c(i) + b(i)*b(i) + c(i)
      d(i) = a(i) + a_old(i+1)
    end do
    !$omp end parallel do

    deallocate(a_old)
  end subroutine tsvc_2_s1244_fp64
end module tsvc_2_s1244_mod
