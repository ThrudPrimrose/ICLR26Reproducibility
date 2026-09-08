module tsvc_2_s3111_mod
  use iso_c_binding, only: c_int64_t, c_double
  implicit none
contains
  subroutine tsvc_2_s3111_fp64(a, b, LEN_1D) bind(C, name="tsvc_2_s3111_fp64")
    implicit none
    integer(c_int64_t), value :: LEN_1D
    real(c_double), dimension(*), intent(in) :: a
    real(c_double), dimension(*), intent(out) :: b
    real(c_double) :: sum
    integer(c_int64_t) :: i

    sum = 0.0_c_double
    !$omp parallel do reduction(+:sum) default(none) shared(a, LEN_1D) private(i)
    do i = 1, LEN_1D
      if (a(i) > 0.0_c_double) then
        sum = sum + a(i)
      end if
    end do
    !$omp end parallel do

    b(1) = sum
  end subroutine tsvc_2_s3111_fp64
end module tsvc_2_s3111_mod
