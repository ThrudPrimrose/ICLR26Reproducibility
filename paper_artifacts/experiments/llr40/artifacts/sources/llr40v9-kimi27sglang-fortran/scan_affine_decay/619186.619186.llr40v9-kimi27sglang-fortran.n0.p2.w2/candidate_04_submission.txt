subroutine scan_affine_decay_fp64(c, x, y, n, workspace, wbytes) &
     bind(c, name="scan_affine_decay_fp64")
  use iso_c_binding, only: c_double, c_int64_t, c_ptr
  use omp_lib
  implicit none
  integer(c_int64_t), value :: n
  type(c_ptr), value :: workspace
  integer(c_int64_t), value :: wbytes
  real(c_double), intent(in) :: c(n), x(n)
  real(c_double), intent(inout) :: y(n)

  integer :: ni, i, b, nt
  integer(c_int64_t) :: n64, lo64, hi64
  real(8) :: acc, prod, seed
  real(8), allocatable :: Cprod(:), Xsum(:), seedX(:)

  ni = int(n)
  if (ni <= 0) return

  if (ni < 1024) then
     y(1) = x(1)
     do i = 2, ni
        y(i) = c(i) * y(i-1) + x(i)
     end do
     return
  end if

  nt = omp_get_max_threads()
  if (nt > ni) nt = ni

  allocate(Cprod(0:nt-1), Xsum(0:nt-1), seedX(0:nt-1))
  n64 = n

  !$omp parallel do schedule(static) private(b, lo64, hi64, i, acc, prod)
  do b = 0, nt - 1
     lo64 = (int(b, c_int64_t) * n64) / int(nt, c_int64_t) + 1_c_int64_t
     hi64 = (int(b + 1, c_int64_t) * n64) / int(nt, c_int64_t)
     acc = 0.0_8
     prod = 1.0_8
     do i = int(lo64), int(hi64)
        prod = prod * c(i)
        acc = c(i) * acc + x(i)
        y(i) = acc
     end do
     Cprod(b) = prod
     Xsum(b) = acc
  end do
  !$omp end parallel do

  seedX(0) = 0.0_8
  do b = 1, nt - 1
     seedX(b) = Cprod(b-1) * seedX(b-1) + Xsum(b-1)
  end do

  !$omp parallel do schedule(static) private(b, lo64, hi64, i, prod, seed)
  do b = 0, nt - 1
     lo64 = (int(b, c_int64_t) * n64) / int(nt, c_int64_t) + 1_c_int64_t
     hi64 = (int(b + 1, c_int64_t) * n64) / int(nt, c_int64_t)
     seed = seedX(b)
     prod = 1.0_8
     do i = int(lo64), int(hi64)
        prod = prod * c(i)
        y(i) = y(i) + prod * seed
     end do
  end do
  !$omp end parallel do

  deallocate(Cprod, Xsum, seedX)
end subroutine scan_affine_decay_fp64
