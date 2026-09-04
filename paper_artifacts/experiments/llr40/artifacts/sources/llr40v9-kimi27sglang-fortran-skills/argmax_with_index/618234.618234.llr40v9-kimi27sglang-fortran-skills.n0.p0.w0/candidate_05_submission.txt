subroutine argmax_with_index_fp64(a, out_index, out_value, LEN_1D) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(in) :: a(LEN_1D)
  integer(c_int64_t), intent(out) :: out_index
  real(c_double), intent(out) :: out_value

  type :: maxloc_t
     real(c_double) :: v
     integer(c_int64_t) :: i
  end type maxloc_t

  !$omp declare reduction(maxloc : maxloc_t : &
  !$omp& omp_out = merge(omp_in, omp_out, &
  !$omp&        omp_in%v > omp_out%v .or. &
  !$omp&        (omp_in%v == omp_out%v .and. omp_in%i < omp_out%i))) &
  !$omp& initializer(omp_priv = maxloc_t(-huge(0.0_c_double), huge(1_c_int64_t)))

  type(maxloc_t) :: res
  integer(c_int64_t) :: i

  res = maxloc_t(a(1), 1_c_int64_t)
  !$omp parallel do reduction(maxloc : res) schedule(static)
  do i = 2, LEN_1D
     if (a(i) > res%v .or. (a(i) == res%v .and. i < res%i)) then
        res%v = a(i)
        res%i = i
     end if
  end do

  if (res%i > LEN_1D) res = maxloc_t(a(1), 1_c_int64_t)

  out_value = res%v
  out_index = res%i
end subroutine argmax_with_index_fp64
