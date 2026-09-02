module fuse_move_ifs_m
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
contains
  subroutine fuse_move_ifs_fp64(a, b, cond, src, K, LEN_2D) bind(c, name="fuse_move_ifs_fp64")
    implicit none
    integer(c_int64_t), value, intent(in) :: K, LEN_2D
    real(c_double), intent(inout) :: a(LEN_2D,LEN_2D), b(LEN_2D,LEN_2D)
    real(c_double), intent(in) :: cond(LEN_2D), src(LEN_2D,LEN_2D)
    integer(c_int64_t) :: i, j, n
    n = LEN_2D
    !$omp parallel private(i,j)
    if (K > 0_c_int64_t) then
      !$omp do schedule(nonmonotonic:guided) nowait
      do i = 1_c_int64_t, n
        if (cond(i) > 0.0_c_double) then
          !$omp simd
          do j = 1_c_int64_t, n
            a(j,i) = src(j,i) * 2.0_c_double
          end do
          !$omp simd
          do j = 1_c_int64_t, n
            b(j,i) = src(j,i) + 1.0_c_double
          end do
        else
          !$omp simd
          do j = 1_c_int64_t, n
            b(j,i) = src(j,i) + 1.0_c_double
          end do
        end if
      end do
      !$omp end do
    else
      !$omp do schedule(nonmonotonic:guided) nowait
      do i = 1_c_int64_t, n
        if (cond(i) > 0.0_c_double) then
          !$omp simd
          do j = 1_c_int64_t, n
            a(j,i) = src(j,i) * 2.0_c_double
          end do
        end if
      end do
      !$omp end do
    end if
    !$omp end parallel
  end subroutine fuse_move_ifs_fp64
end module fuse_move_ifs_m
