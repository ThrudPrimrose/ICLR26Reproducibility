module tsvc_2_s255_mod
  implicit none
contains

  subroutine s255(a, b, LEN_1D)
    integer, intent(in) :: LEN_1D
    real(8), intent(in) :: b(LEN_1D)
    real(8), intent(out) :: a(LEN_1D)
    integer :: i, prev1, prev2
    real(8), parameter :: inv3 = 0.333d0
    !$omp parallel do default(none) shared(a,b,LEN_1D) private(i,prev1,prev2) schedule(static)
    do i = 1, LEN_1D
       prev1 = i - 1
       if (prev1 < 1) prev1 = prev1 + LEN_1D
       prev2 = i - 2
       if (prev2 < 1) prev2 = prev2 + LEN_1D
       a(i) = (b(i) + b(prev1) + b(prev2)) * inv3
    end do
    !$omp end parallel do
  end subroutine s255

  subroutine tsvc_2_s255_fp64(a, b, n, ws, ws_bytes) bind(C, name="tsvc_2_s255_fp64")
    use iso_c_binding
    type(c_ptr), value :: ws
    integer(c_int64_t), value :: ws_bytes
    integer(c_int64_t), value :: n
    real(c_double), intent(out) :: a(*)
    real(c_double), intent(in) :: b(*)
    integer :: len
    len = n
    call s255(a, b, len)
  end subroutine tsvc_2_s255_fp64

end module tsvc_2_s255_mod
