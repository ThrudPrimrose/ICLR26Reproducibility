subroutine tsvc_2_s119_fp64(aa, bb, len_2d, ws, ws_bytes) bind(C, name="tsvc_2_s119_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value    :: len_2d
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(in)    :: bb(len_2d, len_2d)
  type(c_ptr), intent(in)       :: ws
  integer(c_int64_t), value     :: ws_bytes

  integer(c_int64_t) :: n, j, k

  n = len_2d
  if (n < 2) return

  do j = 2, n
    !$omp simd
    do k = 1, n-1
      aa(k+1, j) = aa(k, j-1) + bb(k+1, j)
    end do
  end do
end subroutine tsvc_2_s119_fp64
