! TSVC s311: sum reduction. sum_out(1) = sum(a(1:LEN_1D)).
! gfortran vectorizes the reduction loop to AVX-512 (full width) and OpenMP
! spreads contiguous chunks across threads.
subroutine tsvc_2_s311_fp64(a, sum_out, LEN_1D, workspace, workspace_size) &
     bind(C, name="tsvc_2_s311_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), intent(in) :: a(LEN_1D)
  real(c_double), intent(inout) :: sum_out(LEN_1D)
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)

  integer(c_int64_t) :: n, i
  real(c_double) :: total

  n = LEN_1D
  total = 0.0d0

  if (n <= 262144) then
     do i = 1, n
        total = total + a(i)
     end do
  else
     !$omp parallel do default(none) shared(a, n) private(i) reduction(+:total) schedule(static, 16384)
     do i = 1, n
        total = total + a(i)
     end do
     !$omp end parallel do
  end if

  sum_out(1) = total
end subroutine tsvc_2_s311_fp64
