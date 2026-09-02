subroutine wf_north_west_fp64(a, len_2d, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d, workspace_size
  real(c_double), intent(inout) :: a(len_2d, len_2d)
  real(c_double), intent(inout) :: workspace(workspace_size)
  integer :: n, d, r, c, p, q, s, il, ih, jl, jh, ip, iq, nB, B, rlo, rhi
  n = int(len_2d)
  if (n < 2) return
  B = 64
  nB = (n - 1 + B - 1) / B
  !$omp parallel
  do d = 0, 2*nB - 2
    rlo = max(0, d - nB + 1)
    rhi = min(nB - 1, d)
    !$omp do
    do r = rlo, rhi
      c = d - r
      jl = 2 + r*B
      jh = min(n, jl + B - 1)
      il = 2 + c*B
      ih = min(n, il + B - 1)
      ip = jh - jl + 1
      iq = ih - il + 1
      do q = 1, iq
        a(jl, il+q-1) = a(jl, il+q-1) + a(jl, il+q-2) + a(jl-1, il+q-1)
      end do
      do p = 2, ip
        a(jl+p-1, il) = a(jl+p-1, il) + a(jl+p-1, il-1) + a(jl+p-2, il)
      end do
      do s = 4, ip + iq
        do p = max(2, s-iq), min(ip, s-2)
          a(jl+p-1, il+s-p-1) = a(jl+p-1, il+s-p-1) + a(jl+p-1, il+s-p-2) + a(jl+p-2, il+s-p-1)
        end do
      end do
    end do
  end do
  !$omp end parallel
end subroutine wf_north_west_fp64
