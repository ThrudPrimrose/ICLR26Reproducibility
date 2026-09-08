subroutine tsvc_2_s2233_fp64(aa, bb, cc, len_2d) bind(C)
  use iso_c_binding
  use omp_lib
  integer(c_int64_t), value, intent(in) :: len_2d
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(inout) :: bb(len_2d, len_2d)
  real(c_double), intent(in)    :: cc(len_2d, len_2d)
  integer(c_int64_t) :: n, b, j, k, nb, wmax, i0, i1, wlen, full
  real(c_double), allocatable :: wa(:), wb(:)

  n = len_2d
  if (n < 9) return
  nb = max(omp_get_max_threads(), 1)
  wmax = (n - 8 + nb - 1) / nb
  full = (n - 8) / 4_c_int64_t
  !$omp parallel do schedule(static) private(wa, wb)
  do b = 1, nb
    i0 = 9 + (b - 1) * wmax
    i1 = min(n, i0 + wmax - 1)
    if (i0 <= n) then
      wlen = i1 - i0 + 1
      allocate(wa(wlen), wb(wlen))
      wa(:) = aa(i0:i1, 8)
      wb(:) = bb(i0:i1, 8)
      do j = 9, 7 + 4*full, 4
        wa(:) = wa(:) + cc(i0:i1, j)
        aa(i0:i1, j) = wa(:)
        wb(:) = wb(:) + cc(i0:i1, j)
        bb(i0:i1, j) = wb(:)
        wa(:) = wa(:) + cc(i0:i1, j + 1)
        aa(i0:i1, j + 1) = wa(:)
        wb(:) = wb(:) + cc(i0:i1, j + 1)
        bb(i0:i1, j + 1) = wb(:)
        wa(:) = wa(:) + cc(i0:i1, j + 2)
        aa(i0:i1, j + 2) = wa(:)
        wb(:) = wb(:) + cc(i0:i1, j + 2)
        bb(i0:i1, j + 2) = wb(:)
        wa(:) = wa(:) + cc(i0:i1, j + 3)
        aa(i0:i1, j + 3) = wa(:)
        wb(:) = wb(:) + cc(i0:i1, j + 3)
        bb(i0:i1, j + 3) = wb(:)
      end do
      do k = 1, mod(n - 8, 4_c_int64_t)
        j = 9 + 4*full + k - 1
        wa(:) = wa(:) + cc(i0:i1, j)
        aa(i0:i1, j) = wa(:)
        wb(:) = wb(:) + cc(i0:i1, j)
        bb(i0:i1, j) = wb(:)
      end do
      deallocate(wa, wb)
    end if
  end do

end subroutine tsvc_2_s2233_fp64
