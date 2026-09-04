subroutine argmax_with_index_fp64(a, out_index, out_value, len_1d) bind(c, name='argmax_with_index_fp64')
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(in) :: a(len_1d)
  integer(c_int64_t), intent(out) :: out_index(1)
  real(c_double), intent(out) :: out_value(1)

  integer, parameter :: V = 8
  integer(c_int64_t), parameter :: THRESHOLD = 65536_c_int64_t
  real(c_double) :: vmax(V), x, myval, neg_inf
  integer(c_int64_t) :: imax(V), idx, myidx
  integer(c_int64_t) :: nfull, b, j, p

  neg_inf = transfer(-4503599627370496_c_int64_t, 1.0_c_double)
  nfull = len_1d / V

  x = a(1)
  idx = 1_c_int64_t

  if (len_1d < THRESHOLD) then
    do j = 1, V
      vmax(j) = neg_inf
      imax(j) = 1_c_int64_t
    end do

    do b = 1, nfull
      p = (b - 1) * V
!$omp simd
      do j = 1, V
        if (a(p + j) > vmax(j)) then
          vmax(j) = a(p + j)
          imax(j) = p + j
        end if
      end do
    end do

    myval = vmax(1)
    myidx = imax(1)
    do j = 2, V
      if (vmax(j) > myval) then
        myval = vmax(j)
        myidx = imax(j)
      else if (vmax(j) == myval .and. imax(j) < myidx) then
        myidx = imax(j)
      end if
    end do

    if (myval > x) then
      x = myval
      idx = myidx
    else if (myval == x .and. myidx < idx) then
      idx = myidx
    end if
  else
!$omp parallel private(vmax, imax, myval, myidx, b, j, p)
    do j = 1, V
      vmax(j) = neg_inf
      imax(j) = 1_c_int64_t
    end do

!$omp do schedule(static)
    do b = 1, nfull
      p = (b - 1) * V
!$omp simd
      do j = 1, V
        if (a(p + j) > vmax(j)) then
          vmax(j) = a(p + j)
          imax(j) = p + j
        end if
      end do
    end do
!$omp end do

    myval = vmax(1)
    myidx = imax(1)
    do j = 2, V
      if (vmax(j) > myval) then
        myval = vmax(j)
        myidx = imax(j)
      else if (vmax(j) == myval .and. imax(j) < myidx) then
        myidx = imax(j)
      end if
    end do

!$omp critical
    if (myval > x) then
      x = myval
      idx = myidx
    else if (myval == x .and. myidx < idx) then
      idx = myidx
    end if
!$omp end critical
!$omp end parallel
  end if

  do p = nfull * V + 1, len_1d
    if (a(p) > x) then
      x = a(p)
      idx = p
    end if
  end do

  out_value(1) = x
  out_index(1) = idx
end subroutine argmax_with_index_fp64
