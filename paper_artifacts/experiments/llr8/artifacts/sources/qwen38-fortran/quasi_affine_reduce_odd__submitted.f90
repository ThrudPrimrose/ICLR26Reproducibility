subroutine quasi_affine_reduce_odd_fp64(a, out, len_1d, workspace, workspace_size) bind(C, name="quasi_affine_reduce_odd_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d, workspace_size
  real(c_double), intent(in)    :: a(len_1d)
  real(c_double), intent(inout) :: out(1)
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)

  integer(c_int64_t) :: k, b, f, per, kstart, kend
  real(c_double) :: acc, s
  integer(c_int64_t), parameter :: nchunk = 48
  real(c_double) :: partials(0:nchunk-1)

  k = len_1d / 2            ! number of terms: 1-based elements f = 2,4,...,2k

  if (k <= 0) then
     out(1) = 0.0d0
     return
  end if

  per = (k + nchunk - 1) / nchunk

  !$omp parallel do schedule(static)
  do b = 0, nchunk - 1
     kstart = b * per
     if (kstart >= k) then
        partials(b) = 0.0d0
        cycle
     end if
     kend = min(kstart + per, k)
     s = 0.0d0
     !$omp simd reduction(+:s)
     do f = 2 * (kstart + 1), 2 * kend, 2
        s = s + a(f)
     end do
     !$omp end simd
     partials(b) = s
  end do
  !$omp end parallel do

  acc = 0.0d0
  do b = 0, nchunk - 1
     acc = acc + partials(b)
  end do
  out(1) = acc
end subroutine quasi_affine_reduce_odd_fp64
