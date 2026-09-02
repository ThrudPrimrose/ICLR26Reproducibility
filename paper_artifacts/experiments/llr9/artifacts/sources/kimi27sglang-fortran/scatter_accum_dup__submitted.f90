module scatter_accum_dup_mod
  use, intrinsic :: iso_c_binding, only: c_double, c_int32_t, c_int64_t, c_char
  implicit none
contains
  subroutine scatter_accum_dup_fp64(bins, ip, src, LEN_1D, workspace, workspace_bytes) bind(C, name='scatter_accum_dup_fp64')
    real(c_double), intent(inout) :: bins(*)
    integer(c_int32_t), intent(in) :: ip(*)
    real(c_double), intent(in) :: src(*)
    integer(c_int64_t), value :: LEN_1D
    character(c_char), intent(inout) :: workspace(*)
    integer(c_int64_t), value :: workspace_bytes

    integer(c_int64_t) :: i, idx1, idx2, idx3, idx4
    integer(c_int64_t) :: last

    last = LEN_1D - mod(LEN_1D, 4_c_int64_t)

    !$omp parallel do private(i, idx1, idx2, idx3, idx4)
    do i = 1, last, 4
      idx1 = int(ip(i), c_int64_t)
      idx2 = int(ip(i+1), c_int64_t)
      idx3 = int(ip(i+2), c_int64_t)
      idx4 = int(ip(i+3), c_int64_t)
      !$omp atomic update
      bins(idx1) = bins(idx1) + src(i)
      !$omp end atomic
      !$omp atomic update
      bins(idx2) = bins(idx2) + src(i+1)
      !$omp end atomic
      !$omp atomic update
      bins(idx3) = bins(idx3) + src(i+2)
      !$omp end atomic
      !$omp atomic update
      bins(idx4) = bins(idx4) + src(i+3)
      !$omp end atomic
    end do
    !$omp end parallel do

    do i = last + 1, LEN_1D
      idx1 = int(ip(i), c_int64_t)
      !$omp atomic update
      bins(idx1) = bins(idx1) + src(i)
      !$omp end atomic
    end do
  end subroutine scatter_accum_dup_fp64
end module scatter_accum_dup_mod
