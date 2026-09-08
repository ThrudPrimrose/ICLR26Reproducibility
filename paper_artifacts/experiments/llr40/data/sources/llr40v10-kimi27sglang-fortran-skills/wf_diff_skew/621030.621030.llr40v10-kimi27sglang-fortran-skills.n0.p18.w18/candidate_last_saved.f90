subroutine wf_diff_skew_fp64(a, LEN_2D, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), intent(inout) :: a(0:*)
  integer(c_int64_t) :: n, nr, nc, B, rt, ct, d, tr, tc, rs, re, cs, ce, r, c, base

  n = LEN_2D
  if (n < 2) return
  nr = n - 1
  nc = n - 1
  B = 128
  rt = (nr + B - 1) / B
  ct = (nc + B - 1) / B

  do d = 2, rt + ct
    !$omp parallel do schedule(static, 1) private(tr, tc, rs, re, cs, ce, r, c, base)
    do tr = max(1_c_int64_t, d - ct), min(rt, d - 1)
      tc = d - tr
      rs = 2 + (tr - 1) * B
      re = min(n, 1 + tr * B)
      cs = 1 + (tc - 1) * B
      ce = min(n - 1, tc * B)
      do r = rs, re
        base = (r - 1) * n + (cs - 1)
        !$omp simd
        do c = cs, ce
          a(base) = a(base) + a(base - n) + a(base - n + 1)
          base = base + 1
        end do
        !$omp end simd
      end do
    end do
    !$omp end parallel do
  end do
end subroutine wf_diff_skew_fp64
