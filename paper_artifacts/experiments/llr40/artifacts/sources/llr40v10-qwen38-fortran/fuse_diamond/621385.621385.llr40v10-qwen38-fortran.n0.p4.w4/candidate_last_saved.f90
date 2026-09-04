subroutine fuse_diamond_fp64(a, out, LEN_1D) bind(C, name='fuse_diamond_fp64')
  use, intrinsic :: iso_c_binding
  implicit none
  type(c_ptr), value, intent(in) :: a
  type(c_ptr), value, intent(in) :: out
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), dimension(:), pointer :: ap, op
  integer(c_int64_t) :: n, i8

  n = LEN_1D
  if (n <= 0) return
  call c_f_pointer(a, ap, shape=[n])
  call c_f_pointer(out, op, shape=[n])

  if (n >= 65536_8) then
    !$omp parallel do default(none) shared(ap, op, n) schedule(static)
    do i8 = 1_8, n
      op(i8) = (ap(i8) * ap(i8) + 1.0d0) * (ap(i8) * ap(i8) - 1.0d0)
    end do
    !$omp end parallel do
  else
    do i8 = 1_8, n
      op(i8) = (ap(i8) * ap(i8) + 1.0d0) * (ap(i8) * ap(i8) - 1.0d0)
    end do
  end if
end subroutine
