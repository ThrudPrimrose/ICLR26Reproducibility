module compact_threshold_pack_mod
  use iso_c_binding
  use omp_lib
  implicit none
contains
  subroutine compact_threshold_pack_fp64(out_count, src, weight, packed, len_1d, workspace, workspace_size) &
    bind(C, name="compact_threshold_pack_fp64")
    integer(c_int64_t), intent(out) :: out_count(1)
    integer(c_int64_t), value, intent(in) :: len_1d
    type(c_ptr), value, intent(in) :: workspace
    integer(c_int64_t), value, intent(in) :: workspace_size
    real(c_double), intent(in) :: src(len_1d)
    real(c_double), intent(in) :: weight(len_1d)
    real(c_double), intent(inout) :: packed(len_1d)

    integer(c_int64_t) :: i, n
    integer(c_int64_t), pointer :: prefix(:)
    ! Use workspace as temporary prefix array if large enough.
    logical :: allocated_locally
    allocated_locally = .false.
    if (.not. c_associated(workspace, c_null_ptr) .and. workspace_size >= len_1d * 8_c_int64_t) then
        call c_f_pointer(workspace, prefix, [len_1d])
    else
        allocate(prefix(len_1d))
        allocated_locally = .true.
    end if

    ! First pass: compute inclusive prefix sum of mask (src > 0).
    n = 0_c_int64_t
    do i = 1, len_1d
        if (src(i) > 0.0_c_double) then
            n = n + 1_c_int64_t
            prefix(i) = n
        else
            prefix(i) = n
        end if
    end do
    out_count(1) = n

    ! Second pass: write packed values in parallel using prefix indices.
    if (n > 0_c_int64_t) then
        !$omp parallel do default(none) shared(src,weight,packed,prefix,len_1d) private(i)
        do i = 1, len_1d
            if (src(i) > 0.0_c_double) then
                packed(prefix(i)) = src(i) * weight(i)
            end if
        end do
        !$omp end parallel do
    end if

    ! Deallocate prefix if we allocated it locally.
    if (allocated_locally) then
        deallocate(prefix)
    end if
  end subroutine compact_threshold_pack_fp64
end module compact_threshold_pack_mod
