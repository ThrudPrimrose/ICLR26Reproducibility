subroutine tsvc_2_s318_fp64(a, result, LEN_1D, inc) bind(C, name="tsvc_2_s318_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value :: LEN_1D, inc
  real(c_double), dimension(*), intent(in) :: a
  real(c_double), dimension(*), intent(out) :: result
  integer(c_int64_t) :: i
  real(c_double) :: maxv, v
  integer(c_int64_t) :: index

  if (LEN_1D <= 0_c_int64_t) then
    result(1) = 0.0_c_double
    return
  end if

  maxv = abs(a(1))

  !$omp parallel do reduction(max:maxv) schedule(static)
  do i = 1_c_int64_t, LEN_1D-1_c_int64_t
    v = abs(a(1 + i*inc))
    if (v > maxv) then
      maxv = v
    end if
  end do
  !$omp end parallel do

  index = -1_c_int64_t
  do i = 0_c_int64_t, LEN_1D-1_c_int64_t
    if (abs(a(1 + i*inc)) == maxv) then
      index = i
      exit
    end if
  end do

  if (index < 0_c_int64_t) index = 0_c_int64_t

  result(1) = maxv + real(index, c_double)
end subroutine tsvc_2_s318_fp64
