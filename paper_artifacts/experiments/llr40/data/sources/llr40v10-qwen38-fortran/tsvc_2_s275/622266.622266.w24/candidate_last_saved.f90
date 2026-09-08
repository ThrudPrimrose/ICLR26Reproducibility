subroutine tsvc_2_s275_fp64(aa, bb, cc, nraw) bind(C, name="tsvc_2_s275_fp64")
  use, intrinsic :: iso_c_binding
  use omp_lib
  implicit none
  type(c_ptr), value :: aa, bb, cc, nraw
  integer(c_int64_t) :: raw
  integer(c_int64_t), pointer :: nderef => null()
  integer :: n, i, j, g, nt, ng, gs, lo, hi
  real(c_double), pointer :: a1d(:), b1d(:), c1d(:)
  real(c_double), allocatable :: t(:)
  logical, allocatable :: m(:)

  raw = transfer(nraw, raw)
  if (raw < 2_c_int64_t**30) then
     n = int(raw)
  else
     call c_f_pointer(nraw, nderef)
     n = int(nderef)
  end if

  allocate(t(n), m(n))
  call c_f_pointer(aa, a1d, [n * n])
  call c_f_pointer(bb, b1d, [n * n])
  call c_f_pointer(cc, c1d, [n * n])

  m(1:n) = a1d(1:n) > 0.0d0

  nt = omp_get_max_threads()
  gs = 128
  if (gs > n) gs = max(1, n)
  ng = (n + gs - 1) / gs

  !$omp parallel do schedule(static)
  do g = 0, ng - 1
     lo = g * gs + 1
     hi = min(n, (g + 1) * gs)
     do j = 1, n - 1
        do i = lo, hi
           t(i) = b1d(j * n + i) * c1d(j * n + i)
        end do
        do i = lo, hi
           a1d(j * n + i) = merge(a1d((j - 1) * n + i) + t(i), a1d(j * n + i), m(i))
        end do
     end do
  end do
end subroutine tsvc_2_s275_fp64
