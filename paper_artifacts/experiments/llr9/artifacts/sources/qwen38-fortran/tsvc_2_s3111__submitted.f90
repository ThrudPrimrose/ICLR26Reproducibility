subroutine tsvc_2_s3111_fp64(a, b, len_1d, ws, ws_bytes) bind(c, name="tsvc_2_s3111_fp64")
  use iso_c_binding
  implicit none
  real(c_double), intent(in)  :: a(*)
  real(c_double), intent(out) :: b(*)
  integer(c_int64_t), value, intent(in) :: len_1d
  type(c_ptr), value, intent(in) :: ws
  integer(c_int64_t), value, intent(in) :: ws_bytes
  integer(c_int64_t) :: i
  real(c_double) :: s, ai

  s = 0.0d0
  !$omp parallel private(i, ai) reduction(+:s) num_threads(48)
  !$omp do
  do i = 1, len_1d
     ai = a(i)
     if (ai > 0.0d0) s = s + ai
  end do
  !$omp end do
  !$omp end parallel
  b(1) = s
end subroutine tsvc_2_s3111_fp64
