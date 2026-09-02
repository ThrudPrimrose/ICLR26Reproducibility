subroutine wavefront2d_fp64(a, len_2d, workspace, workspace_size) bind(C, name="wavefront2d_fp64")
  use, intrinsic :: iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d, workspace_size
  real(c_double), intent(inout) :: a(len_2d, len_2d)
  type(c_ptr), value, intent(in) :: workspace

  integer :: n, m, b, nb
  integer :: w, bi, bj, rmin, rmax, cmin, cmax, s, ulo, uhi, r, c

  n = int(len_2d)
  if (n <= 2) return

  if (n < 512) then
    do c = 2, n
      do r = 2, n
        a(r,c) = 0.25d0 * (((a(r,c) + a(r,c-1)) + a(r-1,c)) + a(r-1,c-1))
      end do
    end do
    return
  end if

  b = 32
  m = n - 1
  nb = (m + b - 1) / b

!$omp parallel
  do w = 0, 2*nb - 2
!$omp barrier
!$omp do schedule(static)
    do bi = max(0, w - nb + 1), min(w, nb - 1)
      bj = w - bi
      rmin = bi*b + 1
      rmax = min((bi+1)*b, m)
      cmin = bj*b + 1
      cmax = min((bj+1)*b, m)
      do s = rmin + cmin, rmax + cmax
        ulo = max(rmin, s - cmax)
        uhi = min(rmax, s - cmin)
        do r = ulo, uhi
          c = s - r
          a(r+1, c+1) = 0.25d0 * (((a(r+1,c+1) + a(r+1,c)) + a(r,c+1)) + a(r,c))
        end do
      end do
    end do
!$omp end do
  end do
!$omp end parallel
end subroutine wavefront2d_fp64
