module argmax_mod
  use, intrinsic :: iso_c_binding, only: c_double, c_int64_t
  implicit none

  type :: pair_t
    real(c_double) :: val
    integer(c_int64_t) :: idx
  end type pair_t

!$omp declare reduction(argmax : pair_t : call argmax_combine(omp_out, omp_in)) &
!$omp& initializer(omp_priv = pair_t(-huge(1.0_c_double), huge(1_c_int64_t)))

contains

  subroutine argmax_combine(pout, pin)
    type(pair_t), intent(inout) :: pout
    type(pair_t), intent(in) :: pin
    if (pin%val > pout%val) then
      pout = pin
    else if (pin%val == pout%val .and. pin%idx < pout%idx) then
      pout%idx = pin%idx
    end if
  end subroutine argmax_combine

end module argmax_mod

subroutine argmax_with_index_fp64(a, out_index, out_value, LEN_1D) bind(c, name="argmax_with_index_fp64")
  use, intrinsic :: iso_c_binding, only: c_double, c_int64_t
  use omp_lib
  use argmax_mod
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(in) :: a(LEN_1D)
  integer(c_int64_t), intent(out) :: out_index(*)
  real(c_double), intent(out) :: out_value(*)

  type(pair_t) :: p
  integer(c_int64_t) :: i

  if (LEN_1D <= 1024_c_int64_t) then
    p = pair_t(a(1), 1_c_int64_t)
    do i = 2_c_int64_t, LEN_1D
      if (a(i) > p%val) then
        p%val = a(i)
        p%idx = i
      end if
    end do
    out_value(1) = p%val
    out_index(1) = p%idx
    return
  end if

  p = pair_t(a(1), 1_c_int64_t)
  !$omp parallel do reduction(argmax : p) schedule(static)
  do i = 2_c_int64_t, LEN_1D
    if (a(i) > p%val) then
      p%val = a(i)
      p%idx = i
    end if
  end do
  !$omp end parallel do

  out_value(1) = p%val
  out_index(1) = p%idx
end subroutine argmax_with_index_fp64
