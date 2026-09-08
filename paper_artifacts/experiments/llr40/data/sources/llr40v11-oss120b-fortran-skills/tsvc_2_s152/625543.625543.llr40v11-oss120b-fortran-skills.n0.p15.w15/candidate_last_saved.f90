module tsvc_2_s152_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s152_fp64(a, b, c, d, e, len_1d) bind(C, name="tsvc_2_s152_fp64")
    implicit none
    integer(c_int64_t), value, intent(in) :: len_1d
    real(c_double), intent(inout) :: a(len_1d)
    real(c_double), intent(inout) :: b(len_1d)
    real(c_double), intent(in) :: c(len_1d), d(len_1d), e(len_1d)
    integer(c_int64_t) :: i
    real(c_double) :: t

    !$omp parallel default(none) shared(a,b,c,d,e,len_1d) private(i, t)
  !$omp do
    do i = 1, len_1d
      t = d(i) * e(i)
      b(i) = t
      a(i) = a(i) + t * c(i)
    end do
    !$omp end do
  !$omp end parallel
  end subroutine tsvc_2_s152_fp64
end module tsvc_2_s152_mod
