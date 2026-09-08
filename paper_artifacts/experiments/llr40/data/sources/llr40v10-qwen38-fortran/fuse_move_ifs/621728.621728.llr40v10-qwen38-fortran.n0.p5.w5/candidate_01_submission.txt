subroutine fuse_move_ifs_fp64(a, b, cond, src, K, LEN_2D) bind(C, name="fuse_move_ifs_fp64")
  use, intrinsic :: iso_c_binding
  implicit none
  type(c_ptr), value, intent(in) :: a, b, cond, src
  integer(c_int64_t), value, intent(in) :: K, LEN_2D

  integer(c_int64_t) :: n, nn, i, base, j
  real(c_double), dimension(:), pointer :: ap, bp, sp, cp

  n = LEN_2D
  nn = n*n
  call c_f_pointer(a, ap, [nn])
  call c_f_pointer(b, bp, [nn])
  call c_f_pointer(src, sp, [nn])
  call c_f_pointer(cond, cp, [n])

  !$omp parallel do
  do i = 1, n
     if (cp(i) > 0.0d0) then
        base = (i - 1) * n
        do j = 1, n
           ap(base + j) = 2.0d0 * sp(base + j)
        end do
     end if
  end do

  if (K > 0) then
     !$omp parallel do
     do j = 1, nn
        bp(j) = sp(j) + 1.0d0
     end do
  end if
end subroutine fuse_move_ifs_fp64
