module tsvc_2_s233_mod
  use iso_c_binding
  implicit none
contains

  subroutine tsvc_2_s233_fp64(aa, bb, cc, len_2d, workspace, workspace_size) bind(C, name="tsvc_2_s233_fp64")
    ! Arguments matching the required C ABI.
    integer(c_int64_t), value, intent(in) :: len_2d
    type(c_ptr), value, intent(in) :: workspace          ! unused workspace buffer
    integer(c_int64_t), value, intent(in) :: workspace_size
    real(c_double), intent(inout), dimension(len_2d, len_2d) :: aa
    real(c_double), intent(inout), dimension(len_2d, len_2d) :: bb
    real(c_double), intent(in),    dimension(len_2d, len_2d) :: cc
    integer(c_int64_t) :: i, j

    ! -----------------------------------------------------------------
    ! First phase: update aa. Recurrence is across rows (second index).
    ! The columns (first index) are independent, so we parallelise over i.
    ! -----------------------------------------------------------------
    !$omp parallel do schedule(static) private(j)
    do i = 9_c_int64_t, len_2d
      do j = 9_c_int64_t, len_2d
        aa(i, j) = aa(i, j - 1) + cc(i, j)
      end do
    end do
    !$omp end parallel do

    ! -----------------------------------------------------------------
    ! Second phase: update bb. Recurrence is across columns (first index).
    ! The rows (second index) are independent, so we parallelise over j.
    ! -----------------------------------------------------------------
    !$omp parallel do schedule(static) private(i)
    do j = 9_c_int64_t, len_2d
      do i = 9_c_int64_t, len_2d
        bb(i, j) = bb(i - 1, j) + cc(i, j)
      end do
    end do
    !$omp end parallel do

  end subroutine tsvc_2_s233_fp64

end module tsvc_2_s233_mod
