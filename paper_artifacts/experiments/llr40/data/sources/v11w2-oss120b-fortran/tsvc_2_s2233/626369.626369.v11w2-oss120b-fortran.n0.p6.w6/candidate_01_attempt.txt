module tsvc_2_s2233_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s2233_fp64(aa, bb, cc, LEN_2D) bind(C, name="tsvc_2_s2233_fp64")
    integer(c_int64_t), value :: LEN_2D
    real(c_double), intent(inout) :: aa(*)
    real(c_double), intent(inout) :: bb(*)
    real(c_double), intent(in)    :: cc(*)
    integer(c_int64_t) :: i, j
    integer(c_int64_t) :: idx_curr, idx_prev
    !$omp parallel private(i, j, idx_curr, idx_prev)
    !$omp do schedule(static) nowait
    do i = 9, LEN_2D
      do j = 9, LEN_2D
        idx_prev = (j-2) * LEN_2D + i   ! aa[(j-1)*LEN_2D + i]
        idx_curr = (j-1) * LEN_2D + i   ! aa[j*LEN_2D + i]
        aa(idx_curr) = aa(idx_prev) + cc(idx_curr)
      end do
    end do
    !$omp end do
    !$omp do schedule(static) nowait
    do j = 9, LEN_2D
      do i = 9, LEN_2D
        idx_prev = (i-2) * LEN_2D + j   ! bb[(i-1)*LEN_2D + j]
        idx_curr = (i-1) * LEN_2D + j   ! bb[i*LEN_2D + j]
        bb(idx_curr) = bb(idx_prev) + cc(idx_curr)
      end do
    end do
    !$omp end do
    !$omp end parallel
  end subroutine tsvc_2_s2233_fp64
end module tsvc_2_s2233_mod
