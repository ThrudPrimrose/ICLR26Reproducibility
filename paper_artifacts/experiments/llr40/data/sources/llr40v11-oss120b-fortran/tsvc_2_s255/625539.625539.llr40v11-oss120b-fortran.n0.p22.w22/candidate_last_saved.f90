module tsvc_2_s255_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s255_fp64(a, b, LEN_1D) bind(C, name='tsvc_2_s255_fp64')
    integer(c_int64_t), value :: LEN_1D
    real(c_double), intent(out) :: a(0:*)
    real(c_double), intent(in) :: b(0:*)
    integer(c_int64_t) :: i
    real(c_double), parameter :: inv3 = 0.333d0
    if (LEN_1D <= 0_c_int64_t) then
      return
    else if (LEN_1D == 1_c_int64_t) then
      a(0) = b(0) * 1.0d0
      return
    else if (LEN_1D == 2_c_int64_t) then
      a(0) = (b(0) + b(1) + b(0)) * inv3
      a(1) = (b(1) + b(0) + b(1)) * inv3
      return
    end if
    a(0) = (b(0) + b(LEN_1D-1) + b(LEN_1D-2)) * inv3
    a(1) = (b(1) + b(0) + b(LEN_1D-1)) * inv3
    !$omp parallel do default(none) shared(b,a,LEN_1D) schedule(static)
    do i = 2, LEN_1D-1
      a(i) = (b(i) + b(i-1) + b(i-2)) * inv3
    end do
    !$omp end parallel do
  end subroutine tsvc_2_s255_fp64
end module tsvc_2_s255_mod
