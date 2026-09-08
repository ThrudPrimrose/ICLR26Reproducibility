subroutine tsvc_2_s316_fp64(a, result, len_1d) bind(C, name="tsvc_2_s316_fp64")
  use, intrinsic :: iso_c_binding
  use, intrinsic :: omp_lib
  implicit none
  type(c_ptr), value :: a
  type(c_ptr), value :: result
  integer(c_int64_t), value :: len_1d

  real(c_double), pointer :: av(:), rv(:)
  real(c_double) :: x
  integer(kind=8) :: n, lo, hi, nc
  integer(kind=8) :: c
  integer :: nt

  n = len_1d
  if (n < 1) return
  call c_f_pointer(a, av, [n])
  call c_f_pointer(result, rv, [1])

  x = av(1)
  nt = 32
  if (n < 4000000) then
     x = minval(av)
  else
     nc = (n + 999999) / 1000000
!$omp parallel shared(av,n,nc) reduction(min:x) num_threads(nt)
        x = 1.0d308
!$omp do
        do c = 0, nc - 1
           lo = 1
           if (c > 0) lo = int(c, 8) * 1000000 + 1
           hi = n
           if (lo < n) hi = min(n, lo + 999999)
           if (lo <= hi) x = min(x, fmin2(av, lo, hi))
        end do
!$omp end do
!$omp end parallel
  end if
  rv(1) = x
contains
  pure function fmin2(arr, l, h) result(r)
    real(c_double), intent(in) :: arr(:)
    integer(kind=8), intent(in) :: l, h
    real(c_double) :: r
    integer(kind=8) :: i, j
    r = arr(l)
    i = l
    do while (i + 16 <= h)
       r = min(r, minval(arr(i:i+7)), minval(arr(i+8:i+15)))
       i = i + 16
    end do
    do j = i, h
       if (arr(j) < r) r = arr(j)
    end do
  end function fmin2
end subroutine tsvc_2_s316_fp64
