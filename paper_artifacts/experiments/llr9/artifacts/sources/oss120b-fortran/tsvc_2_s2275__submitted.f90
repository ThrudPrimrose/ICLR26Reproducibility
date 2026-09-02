module tsvc_2_s2275_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s2275_fp64(a, aa, b, bb, c, cc, d, LEN_2D) bind(C, name="tsvc_2_s2275_fp64")
    use iso_c_binding
    implicit none
    integer(c_int64_t), value :: LEN_2D
    real(c_double), intent(in) :: b(*), bb(*), c(*), cc(*), d(*)
    real(c_double), intent(inout) :: a(*), aa(*)
    integer(c_int64_t) :: i, j, idx

    ! Update 2D arrays: loop over columns (j) outer for contiguous rows (i)
    !$omp parallel default(shared) private(j,i,idx)
    !$omp do schedule(static)
    do j = 0_c_int64_t, LEN_2D - 1_c_int64_t
      !$omp simd
      do i = 0_c_int64_t, LEN_2D - 1_c_int64_t
        idx = j * LEN_2D + i
        aa(idx + 1) = aa(idx + 1) + bb(idx + 1) * cc(idx + 1)
      end do
    end do
    !$omp end do

    ! Compute vector a
    !$omp do schedule(static)
    do i = 0_c_int64_t, LEN_2D - 1_c_int64_t
      a(i + 1) = b(i + 1) + c(i + 1) * d(i + 1)
    end do
    !$omp end do
    !$omp end parallel
  end subroutine tsvc_2_s2275_fp64
end module tsvc_2_s2275_mod
