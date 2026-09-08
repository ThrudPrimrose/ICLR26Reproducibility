subroutine compact_threshold_pack_fp64(src, weight, packed, out_count, LEN_1D) bind(C)
    use iso_c_binding, only: c_int64_t, c_double, c_ptr
    use omp_lib
    implicit none
    real(c_double), intent(in) :: src(*)
    real(c_double), intent(in) :: weight(*)
    real(c_double), intent(inout) :: packed(*)
    integer(c_int64_t), intent(out) :: out_count(1)
    integer(c_int64_t), value, intent(in) :: LEN_1D

    integer :: nt, tid, actual_nt
    integer(c_int64_t) :: i
    integer(c_int64_t), allocatable :: thread_sum(:), thread_offset(:)
    integer(c_int64_t) :: start_i, end_i, pos, local_cnt, base_len, rem_len, base_len_local, rem_len_local
    if (LEN_1D <= 0_c_int64_t) then
        out_count(1) = 0_c_int64_t
        return
    end if

    ! Determine number of threads
    nt = omp_get_max_threads()
    base_len = LEN_1D / nt
    rem_len = LEN_1D - base_len * nt
    allocate(thread_sum(nt))
    allocate(thread_offset(nt+1))
    thread_sum = 0_c_int64_t
    thread_offset = 0_c_int64_t

    ! First parallel region: count survivors per thread
    !$omp parallel num_threads(nt) private(tid, i, start_i, end_i, local_cnt, base_len_local, rem_len_local) shared(actual_nt, thread_sum)
    tid = omp_get_thread_num()
    !$omp single
        actual_nt = omp_get_num_threads()
    !$omp end single
    base_len_local = LEN_1D / actual_nt
    rem_len_local = LEN_1D - base_len_local * actual_nt
    if (tid < rem_len_local) then
        start_i = tid * (base_len_local + 1_c_int64_t) + 1_c_int64_t
        end_i = start_i + base_len_local
    else
        start_i = tid * base_len_local + rem_len_local + 1_c_int64_t
        end_i = start_i + base_len_local - 1_c_int64_t
    end if
    local_cnt = 0_c_int64_t
    do i = start_i, end_i
        if (src(i) > 0.0_c_double) then
            local_cnt = local_cnt + 1_c_int64_t
        end if
    end do
    thread_sum(tid+1) = local_cnt
    !$omp end parallel

    ! Compute exclusive prefix sum of thread counts (serial)
    thread_offset(1) = 0_c_int64_t
    do i = 2, actual_nt+1
        thread_offset(i) = thread_offset(i-1) + thread_sum(i-1)
    end do

    ! Second parallel region: write packed values using offsets
    !$omp parallel num_threads(actual_nt) private(tid, i, start_i, end_i, pos, base_len_local, rem_len_local) shared(actual_nt)
    tid = omp_get_thread_num()
    base_len_local = LEN_1D / actual_nt
    rem_len_local = LEN_1D - base_len_local * actual_nt
    if (tid < rem_len_local) then
        start_i = tid * (base_len_local + 1_c_int64_t) + 1_c_int64_t
        end_i = start_i + base_len_local
    else
        start_i = tid * base_len_local + rem_len_local + 1_c_int64_t
        end_i = start_i + base_len_local - 1_c_int64_t
    end if
    pos = thread_offset(tid+1)
    do i = start_i, end_i
        if (src(i) > 0.0_c_double) then
            pos = pos + 1_c_int64_t
            packed(pos) = src(i) * weight(i)
        end if
    end do
    !$omp end parallel

    out_count(1) = thread_offset(actual_nt+1)
    deallocate(thread_sum)
    deallocate(thread_offset)
end subroutine compact_threshold_pack_fp64
