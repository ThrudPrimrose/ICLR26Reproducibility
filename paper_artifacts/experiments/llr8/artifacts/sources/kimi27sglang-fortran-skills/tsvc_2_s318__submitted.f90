subroutine tsvc_2_s318_fp64(a, result, LEN_1D, inc) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D, inc
  real(c_double), intent(in) :: a(0:(LEN_1D-1)*inc)
  real(c_double), intent(inout) :: result(1)

  integer(c_int64_t) :: i, idx, t, nt, lo, hi
  real(c_double) :: maxv, v
  real(c_double), allocatable :: lmax(:)
  integer(c_int64_t), allocatable :: lidx(:)

  if (LEN_1D < 1024) then
    maxv = abs(a(0))
    idx = 0
    do i = 2, LEN_1D
      v = abs(a((i - 1) * inc))
      if (v > maxv) then
        maxv = v
        idx = i - 1
      end if
    end do
    result(1) = maxv + dble(idx)
    return
  end if

  nt = min(int(omp_get_max_threads(), c_int64_t), LEN_1D)
  allocate(lmax(nt), lidx(nt))

  !$omp parallel do private(t, lo, hi, i, v, maxv, idx)
  do t = 1, nt
    lo = ((LEN_1D * (t - 1)) / nt) + 1
    hi = (LEN_1D * t) / nt
    maxv = abs(a((lo - 1) * inc))
    idx = lo - 1
    do i = lo + 1, hi
      v = abs(a((i - 1) * inc))
      if (v > maxv) then
        maxv = v
        idx = i - 1
      end if
    end do
    lmax(t) = maxv
    lidx(t) = idx
  end do

  maxv = lmax(1)
  idx = lidx(1)
  do t = 2, nt
    if (lmax(t) > maxv) then
      maxv = lmax(t)
      idx = lidx(t)
    end if
  end do

  result(1) = maxv + dble(idx)
end subroutine tsvc_2_s318_fp64
